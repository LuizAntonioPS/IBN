package ibn.core;

import java.util.ArrayList;

public class ArrayHelper {
	
	public static int[] toIntArray(ArrayList<Integer> array) {
		int[] list = new int[array.size()];
		for (int i = 0; i < array.size(); i++) {
			list[i] = array.get(i);
		}
		return list;
	}

}
