package ibn;

import java.util.ArrayList;

import ibn.classifiers.bayes.IncrementalBayesNet;
import ibn.classifiers.bayes.net.structure.search.ci.IncrementalST;
import ibn.classifiers.bayes.net.structure.search.local.IncrementalHillClimber;
import ibn.core.BayesNetUtils;
import ibn.core.IncStatistics;
import ibn.core.InstancesHelper;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.core.Instances;

public class Run {

	public static void main(String[] args) throws Exception {
		
		String dataset = InstancesHelper.ALARM_DATASET;
		int initial_structure = BayesNetUtils.EMPTY_MODE;
		int instance_sort = InstancesHelper.RANDOM_MODE;
		int step_size = BayesNetUtils.STEP_SIZE_100;
		int index_algorithm = BayesNetUtils.ST_ALGORITHM;
	
		
		// Algorithm
		SearchAlgorithm algorithm =
				index_algorithm == BayesNetUtils.ST_ALGORITHM ? 
						new IncrementalST() : //new IncrementalST(0.9, 100000);
							new IncrementalHillClimber(1, 2, 2); //new IncrementalHillClimber(100000, Integer.MAX_VALUE-1, Integer.MAX_VALUE-1);
		
		// Initial Structure
		BayesNet bayesNet = BayesNetUtils.getBayesNet(dataset, initial_structure);
		if(bayesNet == null) { return; }
		
		// Data set
		Instances instances = InstancesHelper.getInstances(dataset, instance_sort);
		if(instances == null) { return; }
		
		// Incremental Learning
		IncrementalBayesNet iBayesNet = new IncrementalBayesNet(bayesNet, instances, step_size, algorithm);
		ArrayList<Integer> t = new ArrayList<Integer>();
		t.add(IncStatistics.ACC);
		iBayesNet.setVisualizeMetrics(t);
		iBayesNet.setVisualizeMiddleBayesNets(true);
		iBayesNet.buildStructure();
	}
}
