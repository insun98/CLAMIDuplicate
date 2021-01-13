package edu.handong.csee.ISEL.CLAMI.util;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.commons.math3.stat.StatUtils;

import com.google.common.primitives.Doubles;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveRange;

public class Utils {
	
	
	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff: cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @return instances labeled by CLA
	 */
	public static void getCLAResult(Instances instances,double percentileCutoff,String positiveLabel,boolean suppress) {
		getCLAResult(instances,percentileCutoff,positiveLabel,suppress,false); // no experimental as default
	}
	
	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @param experimental option to display a result in a line;
	 * @return instances labeled by CLA
	 */
	public static void getCLAResult(Instances instances,double percentileCutoff,String positiveLabel,boolean suppress,boolean experimental) {
		Instances instancesByCLA = getInstancesByCLA(instances, percentileCutoff, positiveLabel);
		
		// Print CLA results
		int TP=0, FP=0,TN=0, FN=0;
		for(int instIdx = 0; instIdx < instancesByCLA.numInstances(); instIdx++){
			if(!suppress)
				System.out.println("CLA: Instance " + (instIdx+1) + " predicted as, " + Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx) +
						", (Actual class: " + Utils.getStringValueOfInstanceLabel(instances,instIdx) + ") ");
			
			// compute T/F/P/N for the original instances labeled.
			if(!Double.isNaN(instances.get(instIdx).classValue())){
				if(Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx).equals(Utils.getStringValueOfInstanceLabel(instances,instIdx))){
					if(Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx).equals(positiveLabel))
						TP++;
					else
						TN++;
				}else{
					if(Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx).equals(positiveLabel))
						FP++;
					else
						FN++;
				}
			}
		}
		
		if (TP+TN+FP+FN>0)
			printEvaluationResult(TP, TN, FP, FN, experimental);
		else if(suppress)
			System.out.println("No labeled instances in the arff file. To see detailed prediction results, try again without the suppress option  (-s,--suppress)");
	}

	/**
	 * Print prediction performance in terms of TP, TN, FP, FN, precision, recall, and f1.
	 * @param tP
	 * @param tN
	 * @param fP
	 * @param fN
	 */
	private static void printEvaluationResult(int tP, int tN, int fP, int fN, boolean experimental) {
		
		double precision = (double)tP/(tP+fP);
		double recall = (double)tP/(tP+fN);
		double f1 = (2*(precision*recall))/(precision+recall);
		
		if(!experimental){
			System.out.println("TP: " + tP);
			System.out.println("FP: " + fP);
			System.out.println("TN: " + tN);
			System.out.println("FN: " + fN);
			
			System.out.println("Precision: " + precision);
			System.out.println("Recall: " + recall);
			System.out.println("F1: " + f1);
		}else{
			System.out.print(precision + "," + recall + "," + f1);
		}
	}

	/**
	 * Get instances labeled by CLA
	 * @param instances
	 * @param percentileCutoff
	 * @param positiveLabel
	 * @return
	 */
	private static Instances getInstancesByCLA(Instances instances, double percentileCutoff, String positiveLabel) {
		
		//System.out.println("\nHigher value cutoff > P" + percentileCutoff );
		
		Instances instancesByCLA = new Instances(instances);
		
		double[] cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(instances, percentileCutoff);
		
		// compute, K = the number of metrics whose values are greater than median, for each instance
		Double[] K = new Double[instances.numInstances()];
		
		for(int instIdx = 0; instIdx < instances.numInstances();instIdx++){
			K[instIdx]=0.0;
			for(int attrIdx = 0; attrIdx < instances.numAttributes();attrIdx++){
				if (attrIdx == instances.classIndex())
					continue;
				
				if(instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]){
					K[instIdx]++;
				}
			}
		}
		
		// compute cutoff for the top half and bottom half clusters
		double cutoffOfKForTopClusters = Utils.getMedian(new ArrayList<Double>(new HashSet<Double>(Arrays.asList(K))));
		
		for(int instIdx = 0; instIdx < instances.numInstances(); instIdx++){
			if(K[instIdx]>cutoffOfKForTopClusters)
				instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
			else
				instancesByCLA.instance(instIdx).setClassValue(getNegLabel(instancesByCLA,positiveLabel));
		}
		return instancesByCLA;
	}

	/**
	 * Get higher value cutoffs for each attribute
	 * @param instances
	 * @param percentileCutoff
	 * @return
	 */
	private static double[] getHigherValueCutoffs(Instances instances, double percentileCutoff) {
		// compute median values for attributes
		double[] cutoffForHigherValuesOfAttribute = new double[instances.numAttributes()];

		for(int attrIdx=0; attrIdx < instances.numAttributes();attrIdx++){
			if (attrIdx == instances.classIndex())
				continue;
			cutoffForHigherValuesOfAttribute[attrIdx] = StatUtils.percentile(instances.attributeToDoubleArray(attrIdx),percentileCutoff);
		}
		return cutoffForHigherValuesOfAttribute;
	}
	
	/**
	 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get instancesByCLA use getCLAResult.
	 * @param testInstances
	 * @param instancesByCLA
	 * @param positiveLabel
	 */
	public static void getCLAMIResult(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff,boolean suppress,String mlAlg) {
		getCLAMIResult(testInstances,instances,positiveLabel,percentileCutoff,suppress,false,mlAlg); //no experimental as default
	}
	
	/**
	 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get instancesByCLA use getCLAResult.
	 * @param testInstances
	 * @param instancesByCLA
	 * @param positiveLabel
	 */
	public static void getCLAMIResult(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff, boolean suppress, boolean experimental, String mlAlg) {
		
		String mlAlgorithm = mlAlg!=null && !mlAlg.equals("")?mlAlg:"weka.classifiers.functions.Logistic";
		
		Instances instancesByCLA = getInstancesByCLA(instances, percentileCutoff, positiveLabel);
		
		// Compute medians
		double[] cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(instancesByCLA,percentileCutoff);
				
		// Metric selection
		
		// (1) get distinct violation scores ordered by ASC
		HashMap<Integer,String> metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(instancesByCLA,cutoffsForHigherValuesOfAttribute,positiveLabel);
		Object[] keys = metricIdxWithTheSameViolationScores.keySet().toArray();
		Arrays.sort(keys);
		
		Instances trainingInstancesByCLAMI = null;
		
		int numberOfInstances=0;
		
		
		// (2) Generate instances for CLAMI. If there are no instances in the first round with the minimum violation scores,
		//     then use the next minimum violation score. (Keys are ordered violation scores)
		Instances newTestInstances = null;
		for(int i=0; i<keys.length; i++){
			int qualifiedViolationFrom=instances.numInstances() - (int)keys[i];
			if((int)keys[keys.length-i-1]>=qualifiedViolationFrom) {
				String highViolationInstances = metricIdxWithTheSameViolationScores.get(keys[keys.length-i-1]);
				if(i>1) {
				for(int k=keys.length-i-1; i<keys.length; k++) {
						highViolationInstances = highViolationInstances+metricIdxWithTheSameViolationScores.get(keys[k]);
					}
				}
				
				
				for(int j = 0; j <highViolationInstances.length(); j++) {
				    if(highViolationInstances.charAt(j) == ',') numberOfInstances++;
				}numberOfInstances+=1;
				String selectedMetricIndices = metricIdxWithTheSameViolationScores.get(keys[i]) + metricIdxWithTheSameViolationScores.get(keys[keys.length-i-1])+ (instancesByCLA.classIndex() +1); // violation 값에 해당하는 metric과 class index 
				trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(instancesByCLA,selectedMetricIndices,true); // 각 MVS에 대하여   metric set +  predicted class index set만 남겻다. 
				newTestInstances = getInstancesByRemovingSpecificAttributes(testInstances,selectedMetricIndices,true); // 각 violation 값에 해당하는 metric 값들과 actual class index값만 남겼다. 
						
				// Instance selection
				cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,percentileCutoff); // get higher value cutoffs from the metric-selected dataset
				String instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,numberOfInstances,cutoffsForHigherValuesOfAttribute,positiveLabel); // violation이 있는 instance만 남긴 set 
				trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,instIndicesNeedToRemove,false); //
			}
			else {
			String selectedMetricIndices = metricIdxWithTheSameViolationScores.get(keys[i]) + (instancesByCLA.classIndex() +1); // violation 값에 해당하는 metric과 class index 
			trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(instancesByCLA,selectedMetricIndices,true); // 각 MVS에 대하여   metric set +  predicted class index set만 남겻다. 
			newTestInstances = getInstancesByRemovingSpecificAttributes(testInstances,selectedMetricIndices,true); // 각 violation 값에 해당하는 metric 값들과 actual class index값만 남겼다. 
					
			// Instance selection
			cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,percentileCutoff); // get higher value cutoffs from the metric-selected dataset
			String instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,0,cutoffsForHigherValuesOfAttribute,positiveLabel); // violation이 있는 instance만 남긴 set 
			trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,instIndicesNeedToRemove,false); // violation이 있는 인스턴스만 지우기 위해 false를 보내서 지운다. 그러면 violation 이 없는 최종 값만 남는다. 
			}
			if(trainingInstancesByCLAMI.numInstances() != 0)
				System.out.println(trainingInstancesByCLAMI.numInstances());// 만약 최종 인스턴스가 1개 이상 일때 더 이상 violation 값을 확인하지 않고 이 트레이 세트를 사용한다. 
				break;
		}
		
		// check if there are no instances in any one of two classes.
		if(trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[0]!=0 &&
				trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[1]!=0){
		
			try {
				Classifier classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
				classifier.buildClassifier(trainingInstancesByCLAMI);
				
				// Print CLAMI results
				int TP=0, FP=0,TN=0, FN=0;
				for(int instIdx = 0; instIdx < newTestInstances.numInstances(); instIdx++){
					double predictedLabelIdx = classifier.classifyInstance(newTestInstances.get(instIdx));
					if(!suppress)
						for(int i=0; i<keys.length; i++){
							metricIdxWithTheSameViolationScores.get(keys[i]);
						}
						System.out.println("CLAMI: Instance " + (instIdx+1) + " predicted as, " + 
							newTestInstances.classAttribute().value((int)predictedLabelIdx)	+
							//((newTestInstances.classAttribute().indexOfValue(positiveLabel))==predictedLabelIdx?"buggy":"clean") +
							", (Actual class: " + Utils.getStringValueOfInstanceLabel(newTestInstances,instIdx) + ") ");
					// compute T/F/P/N for the original instances labeled.
					if(!Double.isNaN(instances.get(instIdx).classValue())){
						if(predictedLabelIdx==instances.get(instIdx).classValue()){
							if(predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
								TP++;
							else
								TN++;
						}else{
							if(predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
								FP++;
							else
								FN++;
						}
					}
				}
				
				Evaluation eval = new Evaluation(trainingInstancesByCLAMI);
				eval.evaluateModel(classifier, newTestInstances);
				
				if (TP+TN+FP+FN>0){
					printEvaluationResult(TP, TN, FP, FN, experimental);
					// print AUC value
					if(!experimental)
						System.out.println("AUC: " + eval.areaUnderROC(newTestInstances.classAttribute().indexOfValue(positiveLabel)));
					else
						System.out.print("," + eval.areaUnderROC(newTestInstances.classAttribute().indexOfValue(positiveLabel)));
				}
				else if(suppress)
					System.out.println("No labeled instances in the arff file. To see detailed prediction results, try again without the suppress option  (-s,--suppress)");
				
			} catch (Exception e) {
				System.err.println("Specify the correct Weka machine learing classifier with a fully qualified name. E.g., weka.classifiers.functions.Logistic");
				e.printStackTrace();
				System.exit(0);
			}
		}else{
			System.err.println("Dataset is not proper to build a CLAMI model! Dataset does not follow the assumption, i.e. the higher metric value, the more bug-prone.");
		}
	}

	private static HashMap<Integer, String> getMetricIndicesWithTheViolationScores(Instances instances,
			double[] cutoffsForHigherValuesOfAttribute, String positiveLabel) {

		int[] violations = new int[instances.numAttributes()];
		
		for(int attrIdx=0; attrIdx < instances.numAttributes(); attrIdx++){
			if(attrIdx == instances.classIndex()){
				violations[attrIdx] = instances.numInstances(); // make this as max to ignore since our concern is minimum violation.
				continue;
			}
			
			for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){
				if (instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)){
						violations[attrIdx]++;
				}else if(instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(getNegLabel(instances, positiveLabel))){
						violations[attrIdx  ]++;
				}
			}
		}
		
		HashMap<Integer,String> metricIndicesWithTheSameViolationScores = new HashMap<Integer,String>();
		
		for(int attrIdx=0; attrIdx < instances.numAttributes(); attrIdx++){
			if(attrIdx == instances.classIndex()){
				continue;
			}
			
			int key = violations[attrIdx];
			
			if(!metricIndicesWithTheSameViolationScores.containsKey(key)){
				metricIndicesWithTheSameViolationScores.put(key,(attrIdx+1) + ",");
			}else{
				String indices = metricIndicesWithTheSameViolationScores.get(key) + (attrIdx+1) + ",";
				metricIndicesWithTheSameViolationScores.put(key,indices);
			}
		}
		
		return metricIndicesWithTheSameViolationScores;
	}

	private static String getSelectedInstances(Instances instances, int numberOfOppositeInstance, double[] cutoffsForHigherValuesOfAttribute,
			String positiveLabel) {
		
		int[] violations = new int[instances.numInstances()]; // 2341
		
		for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){ // 각 인스턴스 
			
			for(int attrIdx=0; attrIdx < instances.numAttributes() - numberOfOppositeInstance; attrIdx++){ // 각 메트릭
				if(attrIdx == instances.classIndex())
					continue; // no need to compute violation score for the class attribute
				
				if (instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx] // 각 메트릭 값이 미디언보다 작거나 같은데 버기로 레이블 되어 있으면 violation ++
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)){
					
						violations[instIdx]++;
				}else if(instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx] // 각 메트릭 값이 미디언보다 큰 클린으로 레이블 되어 있으면 violation ++
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(getNegLabel(instances, positiveLabel))){
						violations[instIdx]++;
				}
			}
		}
		if(numberOfOppositeInstance>0) {
			for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){ // 각 인스턴스 
				
				for(int attrIdx=numberOfOppositeInstance; attrIdx < instances.numAttributes(); attrIdx++){ // 각 메트릭
					if(attrIdx == instances.classIndex())
						continue; // no need to compute violation score for the class attribute
					
					if (instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx] // 각 메트릭 값이 미디언보다 작거나 같은데 버기로 레이블 되어 있으면 violation ++
							&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)){
							violations[instIdx]++;
					}else if(instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx] // 각 메트릭 값이 미디언보다 큰 클린으로 레이블 되어 있으면 violation ++
							&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(getNegLabel(instances, positiveLabel))){
							violations[instIdx]++;
					}
				}
			}
		}
		String selectedInstances = "";
		
		for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){
			if(violations[instIdx]>0)
				selectedInstances += (instIdx+1) + ","; // let the start attribute index be 1 
		}
		
		return selectedInstances;
	}
	
	/**
	 * Get the negative label string value from the positive label value
	 * @param instances
	 * @param positiveLabel
	 * @return
	 */
	static public String getNegLabel(Instances instances, String positiveLabel){
		if(instances.classAttribute().numValues()==2){
			int posIndex = instances.classAttribute().indexOfValue(positiveLabel);
			if(posIndex==0)
				return instances.classAttribute().value(1);
			else
				return instances.classAttribute().value(0);
		}
		else{
			System.err.println("Class labels must be binary");
			System.exit(0);
		}
		return null;
	}
	
	/**
	 * Load Instances from arff file. Last attribute will be set as class attribute
	 * @param path arff file path
	 * @return Instances
	 */
	public static Instances loadArff(String path,String classAttributeName){
		Instances instances=null;
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
			instances = new Instances(reader);
			reader.close();
			instances.setClassIndex(instances.attribute(classAttributeName).index());
		} catch (NullPointerException e) {
			System.err.println("Class label name, " + classAttributeName + ", does not exist! Please, check if the label name is correct.");
			instances = null;
		} catch (FileNotFoundException e) {
			System.err.println("Data file, " +path + ", does not exist. Please, check the path again!");
		} catch (IOException e) {
			System.err.println("I/O error! Please, try again!");
		}

		return instances;
	}
	
	/**
	 * Get label value of an instance
	 * @param instances
	 * @param instance index
	 * @return string label of an instance
	 */
	static public String getStringValueOfInstanceLabel(Instances instances,int intanceIndex){
		return instances.instance(intanceIndex).stringValue(instances.classIndex());
	}
	
	/**
	 * Get median from ArraList<Double>
	 * @param values
	 * @return
	 */
	static public double getMedian(ArrayList<Double> values){
		return getPercentile(values,50);
	}
	
	/**
	 * Get a value in a specific percentile from ArraList<Double>
	 * @param values
	 * @return
	 */
	static public double getPercentile(ArrayList<Double> values,double percentile){
		return StatUtils.percentile(getDoublePrimitive(values),percentile);
	}
	
	/**
	 * Get primitive double form ArrayList<Double>
	 * @param values
	 * @return
	 */
	public static double[] getDoublePrimitive(ArrayList<Double> values) {
		return Doubles.toArray(values);
	}
	
	/**
	 * Get instances by removing specific attributes
	 * @param instances
	 * @param attributeIndices attribute indices (e.g., 1,3,4) first index is 1
	 * @param invertSelection for invert selection, if true, select attributes with attributeIndices bug if false, remote attributes with attributeIndices
	 * @return new instances with specific attributes
	 */
	static public Instances getInstancesByRemovingSpecificAttributes(Instances instances,String attributeIndices,boolean invertSelection){
		Instances newInstances = new Instances(instances);

		Remove remove;

		remove = new Remove();
		remove.setAttributeIndices(attributeIndices);
		remove.setInvertSelection(invertSelection);
		try {
			remove.setInputFormat(newInstances);
			newInstances = Filter.useFilter(newInstances, remove);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}

		return newInstances;
	}
	
	/**
	 * Get instances by removing specific instances
	 * @param instances
	 * @param instance indices (e.g., 1,3,4) first index is 1
	 * @param option for invert selection
	 * @return selected instances
	 */
	static public Instances getInstancesByRemovingSpecificInstances(Instances instances,String instanceIndices,boolean invertSelection){
		Instances newInstances = null;

		RemoveRange instFilter = new RemoveRange();
		instFilter.setInstancesIndices(instanceIndices);
		instFilter.setInvertSelection(invertSelection);

		try {
			instFilter.setInputFormat(instances);
			newInstances = Filter.useFilter(instances, instFilter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newInstances;
	}
}
