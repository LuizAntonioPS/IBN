package ibn.classifiers.bayes;

import java.util.ArrayList;
import java.util.List;

import ibn.core.BayesNetUtils;
import ibn.core.IncStatistics;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.core.Instances;

public class IncrementalBayesNet extends BayesNet {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Initial Bayesian Network.
	 */
	BayesNet bayesNet;

	/**
	 * The datasets headers for the purposes of learning Bayesian Network structures
	 */
	Instances instances;

	/**
	 * The size of learning step
	 */
	int step_size;

	/**
	 * Instances dedicated to training dataset (%)
	 */
	double split_threshold;

	/**
	 * The init index of instances
	 */
	int init_index;
	
	/**
	 * Pops up a GraphVisualizer for the Bayesian networks generated during the incremental learning process 
	 */
	boolean visualizeMiddleBayesNets;
	
	/**
	 * Print a string for the Bayesian networks metrics generated during the incremental learning process 
	 */
	List<Integer> visualizeMetrics = new ArrayList<Integer>();

	public IncrementalBayesNet(BayesNet bayesNet, Instances instances, int step_size, SearchAlgorithm algorithm) {
		setBayesNet(bayesNet);
		setInstances(instances);
		setStepSize(step_size);
		setSearchAlgorithm(algorithm);
	}

	/**
	 * buildStructure determines the network structure/graph of the network. The
	 * default behavior is creating a network where all nodes have the first node as
	 * its parent (i.e., a BayesNet that behaves like a naive Bayes classifier).
	 * This method can be overridden by derived classes to restrict the class of
	 * network structures that are acceptable.
	 * 
	 * @throws Exception in case of an error
	 */
	@Override
	public void buildStructure() throws Exception {
		String dataSet_name = bayesNet.getName();
		if (split_threshold == 0 || split_threshold >= 1) { split_threshold = 0.75; }
		int k = init_index;
		int stop = (int) (instances.size()*split_threshold);
		while ((step_size + k) <= stop) {
			Instances incremental_instances = new Instances(instances);
			incremental_instances.delete();
			incremental_instances.addAll(instances.subList(k, (int) k + step_size));
			getSearchAlgorithm().buildStructure(bayesNet, incremental_instances);
			bayesNet.initCPTs();
			bayesNet.estimateCPTs();
			if(!visualizeMetrics.isEmpty()) { IncStatistics.PrintMetrics(visualizeMetrics, k, bayesNet, instances, stop, dataSet_name); }
			if(visualizeMiddleBayesNets) { BayesNetUtils.visualizeBayesNet(bayesNet, "Learning with " + k + " instances"); }
			k += step_size;
		}
		BayesNetUtils.deleteBIFFFile();
		BayesNetUtils.visualizeBayesNet(bayesNet, "Incremental Final Version");
	}

	public BayesNet getBayesNet() {
		return bayesNet;
	}

	public void setBayesNet(BayesNet bayesNet) {
		this.bayesNet = bayesNet;
	}

	public Instances getInstances() {
		return instances;
	}

	public void setInstances(Instances instances) {
		this.instances = instances;
	}

	public int getSetpSize() {
		return step_size;
	}

	public void setStepSize(int k) {
		this.step_size = k;
	}

	public double getThreshold() {
		return split_threshold;
	}

	public void setThreshold(double split_threshold) {
		this.split_threshold = split_threshold;
	}

	public int getInit_Index() {
		return init_index;
	}

	public void setInit_Index(int init_Index) {
		this.init_index = init_Index;
	}

	public void setVisualizeMiddleBayesNets(boolean visualizeMiddleBayesNets) {
		this.visualizeMiddleBayesNets = visualizeMiddleBayesNets;
	}

	public List<Integer> getVisualizeMetrics() {
		return visualizeMetrics;
	}

	public void setVisualizeMetrics(List<Integer> visualizeMetrics) {
		this.visualizeMetrics = visualizeMetrics;
	}
	
}
