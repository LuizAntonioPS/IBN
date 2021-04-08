package ibn.classifiers.bayes.net.structure.search.ci;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import JavaMI.MutualInformation;
import JavaMI.ProbabilityState;
import ibn.classifiers.bayes.net.structure.search.local.ConstrainedHillClimber;
import ibn.classifiers.bayes.net.structure.search.local.MaximumWeightSpanningTree;
import ibn.core.ArrayHelper;
import ibn.core.BayesNetUtils;
import ibn.core.IncStatistics;
import ibn.core.InstancesHelper;
import ibn.core.InstancesSizeException;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.ParentSet;
import weka.classifiers.bayes.net.search.ci.CISearchAlgorithm;
import weka.classifiers.bayes.net.search.local.HillClimber;
import weka.classifiers.bayes.net.search.local.LocalScoreSearchAlgorithm;
import weka.classifiers.bayes.net.search.local.Scoreable;
import weka.core.Instances;
import weka.core.SelectedTag;

public class IncrementalST extends CISearchAlgorithm {
	
	public IncrementalST() {
		super();
		setInitAsNaiveBayes(false);
	}
	
	public IncrementalST(double alpha, int n_parents) {
		super();
		setInitAsNaiveBayes(false);
		setAlpha(alpha);
		setNParents(n_parents);
	}

	/**
	 *
	 */
	private static final long serialVersionUID = 1L;
	private Instances oldInstances;
	private Instances allInstances;

	/**
	 * Confidence level used to measure the degree of the association between nodes.
	 * 0.9 || 0.99
	 */
	private double m_alpha = 0.9;
	
	/**
	 * Confidence level used to measure the degree of the association between nodes.
	 */
	private int n_parents = 1;

	/**
	 * Instance used to check the conditional independence between nodes
	 */
	private Instances ic_instances;

	/**
	 * Structure to storage the maximum weight spanning tree built using the Chow
	 * and Liu's algorithm
	 */
	private BayesNet mWeightTree;

	/**
	 * Search algorithm used for learning the maximum weight spanning tree.
	 */
	private MaximumWeightSpanningTree searchAlgorithm = new MaximumWeightSpanningTree();

	private BayesNet m_bayesNet;

	List<Integer> lastParents = new ArrayList<Integer>();

	Cache m_Cache = null;
	
	@Override
	protected void search(BayesNet bayesNet, Instances instances) throws Exception {
		List<Integer> parents = new ArrayList<Integer>();
		instances = BayesNetUtils.reorderInstanceIfNedded(bayesNet, instances);
		addInstances(instances);
		this.m_bayesNet = new BayesNet();
		m_bayesNet.m_Instances = new Instances(instances);
		m_bayesNet.initStructure();
		initWeightTree();

		for (int iNode = 0; iNode < maxn(); iNode++) {
			ParentSet oParentSet = m_bayesNet.getParentSet(iNode);
			for (int iParent = 0; iParent < maxn(); iParent++) {
				ic_instances = instances; // used on first independence test (line 5 - call on
														// isConditionalIndependent method)
				if (iNode != iParent && !isConditionalIndependent(iNode, iParent, new int[0], 0)) {
					if (oldInstances.equals(allInstances)) {// remember that every time that two nodes are independents,
															// set true on cache
						if (!hasSeparatorSetBetween(iNode, iParent, allInstances)) {
							if (!heuristicIND(iNode, iParent, bayesNet)) {
								parents.add(iParent);
								oParentSet.addParent(iParent, instances);
								m_Cache.setFalse(iNode, iParent);
								continue;
							}
						}
					} else if (!oldInstances.equals(allInstances)) {
						if (!m_Cache.get(iNode, iParent)) { // if are dependents on lasts
							if (!heuristicIND(iNode, iParent, bayesNet)) {
								parents.add(iParent);
								oParentSet.addParent(iParent, instances);
								m_Cache.setFalse(iNode, iParent);
								continue;
							}
						} else { // if exists independence on lasts
							if (!hasSeparatorSetBetween(iNode, iParent, allInstances)) { // if independence doesn't
																							// maintain on all instances
								if (!heuristicIND(iNode, iParent, bayesNet)) {
									parents.add(iParent);
									oParentSet.addParent(iParent, instances);
									m_Cache.setFalse(iNode, iParent);
									continue;
								}
							}
						}
					}
				}
				m_Cache.setTrue(iNode, iParent);
			}
		}
		bayesNet = hillClimbingSearch(bayesNet);
		this.m_bayesNet = new BayesNet();
	}

	int isDiffParents(List<Integer> parents) {
		if (lastParents.size() != 0) {
			for (Integer parent : parents) {
				if (!lastParents.contains(parent)) {
					lastParents = parents;
					return 1;
				}
			}
		}
		lastParents = parents;
		return 0;
	}

	private BayesNet hillClimbingSearch(BayesNet bayesNet) throws Exception {
		ConstrainedHillClimber climber = new ConstrainedHillClimber();
		climber.setUseArcReversal(true);
		climber.setInitAsNaiveBayes(false);
		climber.setM_bayesNet(m_bayesNet);
		climber.setScoreType(new SelectedTag(Scoreable.BAYES, TAGS_SCORE_TYPE));
		climber.setMaxNrOfParents(n_parents); // By setting it to a value much larger than the number of nodes in the
											// network (the default of 100000 pretty much guarantees this), no
											// restriction on the number of parents is enforced
		bayesNet.setSearchAlgorithm(climber);
		bayesNet.m_Instances = allInstances;
		bayesNet.buildStructure();
		return bayesNet;
	}

	private boolean heuristicIND(int iNode, int iParent, BayesNet bayesNet) throws Exception {
		HashSet<Integer> neighborsiNode = new HashSet<Integer>();
		HashSet<Integer> neighborsiParent = new HashSet<Integer>();

		BayesNetUtils.neighborsOnPath(BayesNetUtils.revert(bayesNet), iNode, iParent);
		neighborsiNode.addAll(new HashSet<Integer>(BayesNetUtils.iHeadNeighbors));
		neighborsiParent.addAll(new HashSet<Integer>(BayesNetUtils.iTailNeighbors));

		BayesNetUtils.neighborsOnPath(mWeightTree, iNode, iParent);
		neighborsiNode.addAll(new HashSet<Integer>(BayesNetUtils.iHeadNeighbors));
		neighborsiParent.addAll(new HashSet<Integer>(BayesNetUtils.iTailNeighbors));

		List<HashSet<Integer>> allNeighbors = new ArrayList<HashSet<Integer>>();
		allNeighbors.add(neighborsiNode);
		allNeighbors.add(neighborsiParent);

		for (HashSet<Integer> neighbors : allNeighbors) {
			if (infoChi(iNode, iParent, neighbors.toArray(new Integer[neighbors.size()])) <= 0) {
				return true;
			}
			while (neighbors.size() > 1) {
				int m = Integer.MAX_VALUE;
				double s_m = Double.MAX_VALUE;
				double s = Double.MAX_VALUE;
				for (int i = 0; i < neighbors.size(); i++) {
					List<Integer> neighborsList = new ArrayList<Integer>(neighbors);
					neighborsList.remove(i);
					double s_i = infoChi(iNode, iParent, neighborsList.toArray(new Integer[neighborsList.size()]));
					if (s_i < s_m) {
						s_m = s_i;
						m = i;
					}
				}
				if (s_m <= 0.0) {
					return true;
				} else if (s_m > s) {
					break; // return to for
				} else {
					s = s_m;
					List<Integer> neighborsList = new ArrayList<Integer>(neighbors);
					neighborsList.remove(m);
					neighbors = new HashSet<Integer>(neighborsList);
				}
			}
			// end while
		}
		return false;
	}

	/**
	 * 
	 * @param iNode
	 * @param iParent
	 * @param iNeighborsNodes
	 * @return
	 */
	private double infoChi(int iNode, int iParent, Integer[] iNeighborsNodes) {
		double[] iNodeValues = allInstances.attributeToDoubleArray(iNode); // start repetition

		int nConfigNode = allInstances.attribute(iNode).numValues();

		double[] iParentValues = allInstances.attributeToDoubleArray(iParent);

		int nConfigParent = allInstances.attribute(iParent).numValues();

		if (iNeighborsNodes.length != 0) {
			double[] iMiddleValues = allInstances.attributeToDoubleArray(iNeighborsNodes[0]);
			int nConfigMiddle = allInstances.attribute(iNeighborsNodes[0]).numValues();

			if (iNeighborsNodes.length > 1) {
				for (int i = 1; i < iNeighborsNodes.length; i++) {
					double[] mergedValues = new double[iMiddleValues.length];
					double[] iMergeMiddleValues = allInstances.attributeToDoubleArray(iNeighborsNodes[i]);
					int nMergeConfigMiddle = allInstances.attribute(iNeighborsNodes[i]).numValues();
					ProbabilityState.mergeArrays(iMiddleValues, iMergeMiddleValues, mergedValues);
					iMiddleValues = mergedValues;
					nConfigMiddle = (nConfigMiddle + nMergeConfigMiddle) / 2;
				}
			} // end repetition

			return 2 * allInstances.numInstances()
					* MutualInformation.calculateConditionalMutualInformation(iNodeValues, iParentValues, iMiddleValues)
					- IncStatistics.upperTailofChiSquaredDistribution(m_alpha,
							(nConfigNode - 1) * (nConfigParent - 1) * nConfigMiddle);
		}

		return 2 * allInstances.numInstances()
				* MutualInformation.calculateMutualInformation(iNodeValues, iParentValues)
				- IncStatistics.upperTailofChiSquaredDistribution(m_alpha, (nConfigNode - 1) * (nConfigParent - 1));
	}

	private boolean hasSeparatorSetBetween(int iNode, int iParent, Instances instances) {
		ic_instances = instances;
		int[] iMiddle = new int[maxn() - 2]; // iMiddle will contain all nodes except iNode and iParent
		int position = 0;
		for (int i = 0; i < maxn(); i++) {
			if (i != iNode && i != iParent) {
				iMiddle[position] = i;
				position++;
			}
		}

		int n = iMiddle.length;

		// This code will find all possible iMiddle subsets to verify the independence
		// between the nodes
		for (int i = 0; i < (1 << n); i++) {
			ArrayList<Integer> iMiddleList = new ArrayList<>();
			int m = 1;
			for (int j = 0; j < n; j++) {
				if ((i & m) > 0) {
					iMiddleList.add(iMiddle[j]);
				}
				m = m << 1;
			}
			int[] iMiddleNodes = ArrayHelper.toIntArray(iMiddleList);
			if (isConditionalIndependent(iNode, iParent, iMiddleNodes, iMiddleNodes.length)) {
				return true;
			}
		}
		return false;
	}

	@Override
	protected boolean isConditionalIndependent(int iNode, int iParent, int[] iMiddleNodes, int nAttributesZ) {
		double[] iNodeValues = ic_instances.attributeToDoubleArray(iNode);

		int nConfigNode = ic_instances.attribute(iNode).numValues();

		double[] iParentValues = ic_instances.attributeToDoubleArray(iParent);

		int nConfigParent = ic_instances.attribute(iParent).numValues();

		if (iMiddleNodes.length != 0) {
			double[] iMiddleValues = ic_instances.attributeToDoubleArray(iMiddleNodes[0]);
			int nConfigMiddle = ic_instances.attribute(iMiddleNodes[0]).numValues();

			if (iMiddleNodes.length > 1) {
				for (int i = 1; i < iMiddleNodes.length; i++) {
					double[] mergedValues = new double[iMiddleValues.length];
					double[] iMergeMiddleValues = ic_instances.attributeToDoubleArray(iMiddleNodes[i]);
					int nMergeConfigMiddle = ic_instances.attribute(iMiddleNodes[i]).numValues();
					ProbabilityState.mergeArrays(iMiddleValues, iMergeMiddleValues, mergedValues);
					iMiddleValues = mergedValues;
					nConfigMiddle = (nConfigMiddle + nMergeConfigMiddle) / 2;
				}
			}

			return 2 * ic_instances.numInstances()
					* MutualInformation.calculateConditionalMutualInformation(iNodeValues, iParentValues,
							iMiddleValues) <= IncStatistics.upperTailofChiSquaredDistribution(m_alpha,
									(nConfigNode - 1) * (nConfigParent - 1) * nConfigMiddle);
		}

		return 2 * ic_instances.numInstances()
				* MutualInformation.calculateMutualInformation(iNodeValues, iParentValues) <= IncStatistics
						.upperTailofChiSquaredDistribution(m_alpha, (nConfigNode - 1) * (nConfigParent - 1));
	}

	/**
	 * Sets whether to init as naive bayes
	 * 
	 * @param bInitAsNaiveBayes whether to init as naive bayes
	 */
	public void setInitAsNaiveBayes(boolean bInitAsNaiveBayes) {
		m_bInitAsNaiveBayes = bInitAsNaiveBayes;
	}

	/**
	 * Sets the confidence level used to measure the degree of the association
	 * between two variable
	 * 
	 * @param alpha confidence level
	 */
	public void setAlpha(double alpha) {
		m_alpha = alpha;
	}

	/**
	 * Gets whether to init as naive bayes
	 *
	 * @return whether to init as naive bayes
	 */
	public boolean getInitAsNaiveBayes() {
		return m_bInitAsNaiveBayes;
	}

	/**
	 * returns the number of attributes
	 * 
	 * @return the number of attributes
	 */
	private int maxn() {
		return allInstances.numAttributes();
	}

	private void addInstances(Instances instances) throws Exception {
		if (allInstances == null && oldInstances == null) {
			allInstances = instances;
			oldInstances = instances;
			initCache();
			return;
		}
		if (instances.numAttributes() != maxn()) {
			throw new InstancesSizeException("Different number of attributes among instances.");
		}
		oldInstances = new Instances(allInstances);
		allInstances.addAll(instances);
	}

	/**
	 * cache for remembering the independence conditional among nodes for
	 * incremental steps
	 */
	class Cache {

		/** change in parent nodes due to conditional independence **/
		boolean[][] independentsNodes;

		/**
		 * c'tor
		 * 
		 * @param nNrOfNodes number of nodes in network, used to determine memory size
		 *                   to reserve
		 */
		Cache(int nNrOfNodes) {
			independentsNodes = new boolean[nNrOfNodes][nNrOfNodes];
			setInitialValues(nNrOfNodes);
		}

		/**
		 * set true to cache entry for the same nodes
		 * 
		 * @param node - checked start node independentParent - independent end node
		 * @return cache value
		 */
		private void setInitialValues(int nNrOfNodes) {
			for (int iNode = 0; iNode < nNrOfNodes; iNode++) {
				setTrue(iNode, iNode);
			}
		}

		/**
		 * set cache entry
		 * 
		 * @param node - checked start node independentParent - independent end node
		 * @return cache value
		 */
		public void setTrue(int node, int independentParent) {
			independentsNodes[node][independentParent] = true;
		} // set true

		/**
		 * set cache entry
		 * 
		 * @param node - checked start node independentParent - independent end node
		 * @return cache value
		 */
		public void setFalse(int node, int independentParent) {
			independentsNodes[node][independentParent] = false;
		} // set true

		/**
		 * get cache entry
		 * 
		 * @param node - start node to check the independence independentParent - end
		 *             node to check the independence
		 * @return cache value
		 */
		public boolean get(int node, int independentParent) {
			return independentsNodes[node][independentParent];
		} // get
	} // class Cache

	/**
	 * initCache initializes the cache
	 * 
	 * @param bayesNet Bayes network to be learned
	 * @param instances data set to learn from
	 * @throws Exception if something goes wrong
	 */
	private void initCache() throws Exception {
		m_Cache = new Cache(maxn());
	} // initCache

	/**
	 * Method to generate the maximum weight spanning tree
	 * 
	 * @throws Exception
	 */
	private void initWeightTree() throws Exception {
		mWeightTree = new BayesNet();
		searchAlgorithm.buildStructure(mWeightTree, allInstances);
	}

	public int getNParents() {
		return n_parents;
	}

	public void setNParents(int n_parents) {
		this.n_parents = n_parents;
	}
}
