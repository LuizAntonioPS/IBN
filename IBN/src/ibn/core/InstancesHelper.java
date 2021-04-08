package ibn.core;

import java.io.IOException;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class InstancesHelper {
	
	public static int RANDOM_MODE = 0;
	public static int SIMILAR_MODE = 1;
	public static int DISSIMILAR_MODE = 2;
	
	/**
	 * [dataset description]
	 * 
	 * Possible incremental configurations
	 * @initial_network Empty
	 * @instance_sort RANDOM || SIMILAR || DISSIMILAR
	 * @step_size 100 || 200 || 400 || 500 || 1000 || 2000 || 4000
	 * @metrics [TODO]
	 */
	public static String ADULT_DATASET = "adult";
	
	/**
	 * [dataset description]
	 * 
	 * Possible incremental configurations
	 * @initial_network EMPTY || PARTIAL
	 * @instance_sort RANDOM || SIMILAR || DISSIMILAR
	 * @step_size 100 || 200 || 400 || 500 || 1000 || 2000 || 4000
	 * @metrics ACC || LL || MDL || MIT ||
	 			PRECISION || RECALL || F_MEASURE ||
				BATCH_EXTRA_EDGES || BATCH_MISSED_EDGES || BATCH_REVERSED_EDGES ||
				STANDARD_EXTRA_EDGES || STANDARD_MISSED_EDGES || STANDARD_REVERSED_EDGES  
	 */
	public static String ALARM_DATASET = "alarm";
	
	/**
	 * [dataset description]
	 * 
	 * Possible incremental configurations
	 * @initial_network EMPTY
	 * @instance_sort RANDOM || SIMILAR || DISSIMILAR
	 * @step_size 100 || 200 || 400 || 500 || 1000 || 2000 || 4000
	 * @metrics [TODO]
	 */
	public static String ASIA_DATASET = "asia";
	
	/**
	 * [dataset description]
	 * 
	 * Possible incremental configurations
	 * @initial_network [TODO]
	 * @instance_sort [TODO]
	 * @step_size [TODO]
	 * @metrics [TODO]
	 */
	public static String CAR_DATASET = "car";
	
	/**
	 * [dataset description]
	 * 
	 * Possible incremental configurations
	 * @initial_network [TODO]
	 * @instance_sort [TODO]
	 * @step_size [TODO]
	 * @metrics [TODO]
	 */
	public static String CHILD_DATASET = "child";
	
	/**
	 * [dataset description]
	 * 
	 * Possible incremental configurations
	 * @initial_network [TODO]
	 * @instance_sort [TODO]
	 * @step_size [TODO]
	 * @metrics [TODO]
	 */
	public static String HAILFINDER_DATASET = "hailfinder";
	
	/**
	 * [dataset description]
	 * 
	 * Possible incremental configurations
	 * @initial_network [TODO]
	 * @instance_sort [TODO]
	 * @step_size [TODO]
	 * @metrics [TODO]
	 */
	public static String INSURANCE_DATASET = "insurance";
	
	/**
	 * [dataset description]
	 * 
	 * Possible incremental configurations
	 * @initial_network [TODO]
	 * @instance_sort [TODO]
	 * @step_size [TODO]
	 * @metrics [TODO]
	 */
	public static String NURSERY_DATASET = "nursery";
	
	public static int nRandom = new Random().nextInt(5000);
	
	public static Instances getInstances(String dataset, int mode) {
		Instances instances = null;
		try {
			DataSource dataSource = new DataSource(String.format("datasets/%s_all.arff", dataset));
			if (mode == RANDOM_MODE) { instances = dataSource.getDataSet(); }
			else if (mode == SIMILAR_MODE) { instances = InstancesHelper.sortInstancesByInstance(dataSource.getDataSet()); }
			else if (mode == DISSIMILAR_MODE) { instances = InstancesHelper.reverseSortInstancesByInstance(dataSource.getDataSet()); }
			else { throw new IllegalArgumentException("mode not found"); }
		} catch (IllegalArgumentException e) {
			System.err.println(e.getMessage());
		} catch (IOException e) { //not catched
			System.err.println("dataset not found");
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		return instances;
	}

	/**
	 * Similar -> Dissimilar
	 * @param instances
	 * @return
	 */
	public static Instances sortInstancesByInstance(Instances instances) {
		//int nRandon = 1000;

		Instance randomInstance = instances.get(nRandom);
		int[] scores = new int[instances.size()];
		Instance[] backup = new Instance[instances.size()];
		int j = 0;
		for (Instance inst : instances) {
			backup[j] = inst;
			scores[j] = calcScore(inst, randomInstance);
			j++;
		}
		
		instances.delete();
		
		instances.add(randomInstance); //2021

		int aux = instances.numAttributes();
		while (aux > -1) {
			for (int i = 0; i < scores.length; i++) {
				if (scores[i] == aux) {
					//System.out.println(scores[i]);
					instances.add(backup[i]);
				}
			}
			aux--;
		}
		return instances;
	}
	
	/**
	 * Dissimilar -> Similar 
	 * @param instances
	 * @return
	 */
	public static Instances reverseSortInstancesByInstance(Instances instances) {
		//int nRandon = 1000;

		Instance randomInstance = instances.get(nRandom);
		int[] scores = new int[instances.size()];
		Instance[] backup = new Instance[instances.size()];
		int j = 0;
		for (Instance inst : instances) {
			backup[j] = inst;
			scores[j] = calcScore(inst, randomInstance);
			j++;
		}
		
		instances.delete();
		
		instances.add(randomInstance); //2021
		
		int aux = -1;
		while (aux <= instances.numAttributes()) {
			for (int i = 0; i < scores.length; i++) {
				if (scores[i] == aux) {
					//System.out.println(scores[i]);
					instances.add(backup[i]);
				}
			}
			aux++;
		}
		return instances;
	}

	private static int calcScore(Instance inst, Instance randomInstance) {
		int score = 0;
		for (int i = 0; i < randomInstance.numValues(); i++) {
			if (randomInstance.value(i) == inst.value(i)) {
				score++;
			}
		}
		return score;
	}

	public static void printScores(Instances instances) {
		int nRandon = 0;
		Instance randomInstance = instances.get(nRandon);
		for (int i = 1; i < instances.size(); i++) {
			System.out.println(calcScore(instances.get(i), randomInstance));
		}
	}

}
