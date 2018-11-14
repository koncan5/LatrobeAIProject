package io;

import java.io.File;
import neuralnetwork.Network;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * A JUnit test class for the IOManager class.
 *
 * @author 18403879 Curtis Alcock
 */
public class IOManagerTest {

    public IOManagerTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    /**
     * Test of readNetwork method, of class IOManager.
     */
    @Test
    public void testReadNetwork_pseudoRandom() {
        System.out.println("readNetwork");
        File file = new File("random.xml");
        int[] neurons = {2, 2, 1};
        long seed = 6969;
        double learningRate = 0.1;
        double momentum = 0;
        Network expResult = new Network(neurons, seed, learningRate, momentum);
        Network result = IOManager.readNetwork(file);
        assertEquals(expResult.toString(), result.toString());
    }

    /**
     * Test of readNetwork method, of class IOManager.
     */
    @Test
    public void testReadNetwork_set() {
        System.out.println("testReadNetwork_set");
        File file = new File("set.xml");
        double[][][] weights = {{{0.5, 0.4}, {0.9, 1.0}}, {{-1.2, 1.1}}};
        double[][] thresholds = {{0.8, -0.1}, {0.3}};
        double momentum = 0.95;
        double learningRate = 0.1;
        Network expResult = new Network(weights, thresholds, learningRate, momentum);
        Network result = IOManager.readNetwork(file);
        assertEquals(expResult.toString(), result.toString());
    }

    /**
     * Test of readTrainingSets method, of class IOManager.
     */
    @Test
    public void testReadTrainingSets() {
        System.out.println("readTrainingSets");
        File file = new File("set.xml");
        double[][] expResult = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
        double[][] result = IOManager.readTrainingSets(file);
        assertArrayEquals(expResult, result);
    }
    
    /**
     * Test of readDesiredOutcomes method, of class IOManager.
     */
    @Test
    public void testReadDesiredOutcomes() {
        System.out.println("readDesiredOutcomes");
        File file = new File("set.xml");
        double[][] expResult = {{0.1},{0.9},{0.1},{0.9}};
        double[][] result = IOManager.readDesiredOutcomes(file);
        assertArrayEquals(expResult, result);
    }

    /**
     * Test of writeNetwork method, of class IOManager.
     */
    @Test
    public void testWriteNetwork() {
        System.out.println("writeNetwork");
        File file = new File("write.xml");
        double[][][] weights = {{{0.5, 0.4}, {0.9, 1.0}}, {{-1.2, 1.1}}};
        double[][] thresholds = {{0.8, -0.1}, {0.3}};
        double momentum = 0;
        double learningRate = 0.1;
        Network instance = new Network(weights, thresholds, learningRate, momentum);
        IOManager.writeNetwork(file, instance);
        String expResult = instance.toString();
        String result = IOManager.readNetwork(file).toString();
        assertEquals(expResult, result);
    }

    /**
     * Test of stringToIntArray method, of class IOManager.
     */
    @Test
    public void testStringToIntArray() {
        System.out.println("stringToIntArray");
        String input = "2,2,1";
        String delimiter = ",";
        int[] expResult = {2,2,1};
        int[] result = IOManager.stringToIntArray(input, delimiter);
        assertArrayEquals(expResult, result);
    }

    /**
     * Test of stringToDoubleArray method, of class IOManager.
     */
    @Test
    public void testStringToDoubleArray_goodinputs() {
        System.out.println("stringToDoubleArray_goodinputs");
        String input = "0,0,1";
        String delimiter = ",";
        double[] expResults = {0.1, 0.1, 0.9};
        double[] result = IOManager.stringToDoubleArray(input, delimiter);
        assertArrayEquals(expResults, result, 0.0);
    }

    /**
     * Test of stringToDoubleArray method, of class IOManager.
     */
    @Test
    public void testStringToDoubleArray_badinputs() {
        System.out.println("stringToDoubleArray_badinputs");
        String input = "2,2,1";
        String delimiter = ",";
        try {
            IOManager.stringToDoubleArray(input, delimiter);
        } catch (IllegalArgumentException e) {
            return;
        }
        fail();
    }
}
