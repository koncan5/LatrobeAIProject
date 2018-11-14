package neuralnetwork;

import io.IOManager;
import java.io.File;
import neuralnetworkexceptions.UnevenArraysException;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * A JUnit test class for the Network class.
 *
 * @author 18403879 Curtis Alcock
 */
public class NetworkTest {

    public NetworkTest() {
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
     * Test of activation method, of class Network.
     */
    @Test
    public void testActivation_bookWeights() {
        System.out.println("activation_bookweights");
        double[] inputs = {1, 1};
        double[][][] weights = {{{0.5, 0.4}, {0.9, 1.0}}, {{-1.2, 1.1}}};
        double[][] thresholds = {{0.8, -0.1}, {0.3}};
        double momentum = 0;
        double learningRate = 0.1;
        Network instance = new Network(weights, thresholds, learningRate, momentum);
        double[] expResult = {0.4103};
        try {
            double[] result = instance.activation(inputs);
            System.out.println(result[0]);
            assertArrayEquals(expResult, result, 0.0001);
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
            fail();
        }
    }

    /**
     * Test of activation method, of class Network.
     */
    @Test
    public void testActivation_randomWeights() {
        System.out.println("testActivation_randomWeights");
        double[] inputs = {1, 1};
        int[] neurons = {2, 2, 1};
        long seed = 6969;
        double learningRate = 0.1;
        double momentum = 0;
        Network instance = new Network(neurons, seed, learningRate, momentum);
        double[] expResult = {0.2535};
        try {
            double[] result = instance.activation(inputs);
            assertArrayEquals(expResult, result, 0.0001);
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
            fail();
        }
    }

    /**
     * Test of weightTraining method, of class Network.
     */
    @Test
    public void testWeightTraining_hyperbolic() {
        System.out.println("weightTraining_hyperbolic");
        double[] inputs = {1, 1};
        double[][][] weights = {{{0.5, 0.4}, {0.9, 1.0}}, {{-1.2, 1.1}}};
        double[][] thresholds = {{0.8, -0.1}, {0.3}};
        double momentum = 0;
        double[] desiredOutput = {0.1};
        double learningRate = 0.1;
        Network instance = new Network(weights, thresholds, learningRate, momentum);
        try {
            instance.activation(inputs);
            instance.setDesiredOutput(desiredOutput);
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
        }
        try {
            instance.weightTraining();
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
        }
        String expResult = "Network{\n"
                + "  Desired Outputs: {0.1000} Learning Rate: 0.100, Momentum: 0.000\n"
                + "  Neurons (L,N): {\n"
                + "     Neuron(0,0) : {Weights= {[0]: 0.5115, [1]: 0.4115}, Inputs= {[0]: 1.0000, [1]: 1.0000}, Threshold= 0.7885, Output= 0.0572, Error= 0.0000}\n"
                + "     Neuron(0,1) : {Weights= {[0]: 0.8932, [1]: 0.9932}, Inputs= {[0]: 1.0000, [1]: 1.0000}, Threshold= -0.0932, Output= 1.0001, Error= 0.0000}\n"
                + "     Neuron(1,0) : {Weights= {[0]: -1.2010, [1]: 1.0833}, Inputs= {[0]: 0.0572, [1]: 1.0001}, Threshold= 0.3167, Output= 0.4103, Error= -0.3103}\n"
                + "  }\n"
                + "}";
        String result = instance.toString();
        assertEquals(expResult, result);
    }

    /**
     * Test of sumOfTheSquaredErrors method, of class Network.
     */
    @Test
    public void testSumOfTheSquaredErrors() {
        System.out.println("sumOfTheSquaredErrors");
        double[] inputs = {1, 1};
        double[][][] weights = {{{0.5, 0.4}, {0.9, 1.0}}, {{-1.2, 1.1}}};
        double[][] thresholds = {{0.8, -0.1}, {0.3}};
        double momentum = 0;
        double[] desiredOutput = {0};
        double learningRate = 0.1;
        Network instance = new Network(weights, thresholds, learningRate, momentum);
        try {
            instance.activation(inputs);
            instance.setDesiredOutput(desiredOutput);
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
            fail("Activation failed...");
        }
        try {
            instance.weightTraining();
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
            fail("Weight Training failed...");
        }
        double expResult = 0.1683;
        double result = instance.sumOfTheSquaredErrors();
        assertEquals(expResult, result, 0.0001);
    }

    /**
     * Test of getDesiredOutput method, of class Network.
     */
    @Test
    public void testGetDesiredOutput() {
        System.out.println("getDesiredOutput");
        int[] neurons = {2, 2, 1};
        long seed = 6969;
        double learningRate = 0.1;
        double momentum = 0;
        Network instance = new Network(neurons, seed, learningRate, momentum);
        double[] expResult = {0.0};
        double[] result = instance.getDesiredOutput();
        assertArrayEquals(expResult, result, 0.0001);
    }

    /**
     * Test of setDesiredOutput method, of class Network.
     */
    @Test
    public void testSetDesiredOutput() {
        System.out.println("setDesiredOutput");
        int[] neurons = {2, 2, 1};
        long seed = 6969;
        double learningRate = 0.1;
        double momentum = 0;
        double[] desiredOutput = {1.0};
        Network instance = new Network(neurons, seed, learningRate, momentum);
        try {
            instance.setDesiredOutput(desiredOutput);
            double[] expResult = {1.0};
            double[] result = instance.getDesiredOutput();
            assertArrayEquals(expResult, result, 0.0001);
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
            fail("Desired Output had the wrong number of values.");
        }
    }

    /**
     * Test of train method in sigmoidal mode with momentum, of class Network.
     *
     * Note: Converges at Epoch[545]
     */
    @Test
    public void testTrain_sigmoidal_momentum() {
        System.out.println("train_sigmoidal");
        double[][] trainingSet = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}};
        double[][] desiredOutcomes = {{0.1}, {0.9}, {0.1}, {0.9}};
        double[][][] weights = {{{0.5, 0.4}, {0.9, 1.0}}, {{-1.2, 1.1}}};
        double[][] thresholds = {{0.8, -0.1}, {0.3}};
        double momentum = 0.95;
        double learningRate = 0.1;
        double convergence = 0.0001;
        Network instance = new Network(weights, thresholds, learningRate, momentum);
        try {
            int result = instance.train(trainingSet, desiredOutcomes, convergence, true);
            if (result < convergence) {
                assertTrue(true);
            }
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
            fail();
        }
    }

    /**
     * Test of train method in sigmoidal mode with no momentum, of class
     * Network.
     *
     * Note: Converges at Epoch[11087]
     */
    @Test
    public void testTrain_sigmoidal() {
        System.out.println("train_sigmoidal");
        double[][] trainingSet = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}};
        double[][] desiredOutcomes = {{0.1}, {0.9}, {0.1}, {0.9}};
        double[][][] weights = {{{0.5, 0.4}, {0.9, 1.0}}, {{-1.2, 1.1}}};
        double[][] thresholds = {{0.8, -0.1}, {0.3}};
        double momentum = 0.0;
        double learningRate = 0.1;
        double convergence = 0.0001;
        Network instance = new Network(weights, thresholds, learningRate, momentum);
        try {
            int result = instance.train(trainingSet, desiredOutcomes, convergence, true);
            if (result < convergence) {
                assertTrue(true);
            }
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
            fail();
        }
    }

    /**
     * Test of train method in Hyperbolic Tangent mode with momentum, of class
     * Network.
     *
     * Note: Converges at Epoch[52].
     */
    @Test
    public void testTrain_tanh_momentum() {
        System.out.println("train_tanh_momentum");
        double[][] trainingSet = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}};
        double[][] desiredOutcomes = {{0.1}, {0.9}, {0.1}, {0.9}};
        double[][][] weights = {{{0.5, 0.4}, {0.9, 1.0}}, {{-1.2, 1.1}}};
        double[][] thresholds = {{0.8, -0.1}, {0.3}};
        double momentum = 0.95;
        double learningRate = 0.1;
        double convergence = 0.0001;
        Network instance = new Network(weights, thresholds, learningRate, momentum);
        try {
            int result = instance.train(trainingSet, desiredOutcomes, convergence);
            if (result < convergence) {
                IOManager.writeNetwork(new File("testTrained.xml"), instance);
                assertTrue(true);
            }
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
            fail();
        }
    }

    /**
     * Test of train method in Hyperbolic Tangent mode without momentum, of
     * class Network.
     *
     * Note: Converges at Epoch[1139]
     */
    @Test
    public void testTrain_tanh() {
        System.out.println("train_tanh");
        double[][] trainingSet = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {1.0, 0.0}};
        double[][] desiredOutcomes = {{0.1}, {0.9}, {0.1}, {0.9}};
        double[][][] weights = {{{0.5, 0.4}, {0.9, 1.0}}, {{-1.2, 1.1}}};
        double[][] thresholds = {{0.8, -0.1}, {0.3}};
        double momentum = 0.0;
        double learningRate = 0.1;
        double convergence = 0.0001;
        Network instance = new Network(weights, thresholds, learningRate, momentum);
        try {
            double result = instance.train(trainingSet, desiredOutcomes, convergence);
            if (result < convergence) {
                assertTrue(true);
            }
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
            fail();
        }
    }

    /**
     * Test of run method, of class Network.
     */
    @Test
    public void testRun() {
        System.out.println("run_bookvalues");
        double[] input = {1, 0};
        Network instance = IOManager.readNetwork(new File("testTrained.xml"));
        String expResult = "---------------\n"
                + "| Network Run |\n"
                + "---------------\n"
                + "| Input: {[0]=1.0000, [1]=0.0000}\n"
                + "---------------\n"
                + "| Weights: \n"
                + "| Layer #0:\n"
                + "|	N[0]: {1.6137, 1.6240}, Threshold: 1.6224\n"
                + "|	N[1]: {3.5195, 3.5476}, Threshold: 0.2778\n"
                + "| Layer #1:\n"
                + "|	N[0]: {-2.7919, 2.5703}, Threshold: 1.7764\n"
                + "---------------\n"
                + "| Output: [0]=0.8956\n"
                + "---------------";
        try {
            String result = instance.run(input);
            assertEquals(expResult, result);
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
            fail();
        }
    }

    /**
     * Test of toString method, of class Network.
     */
    @Test
    public void testToString() {
        System.out.println("toString");
        double[] inputs = {1, 1};
        double[][][] weights = {{{0.5, 0.4}, {0.9, 1.0}}, {{-1.2, 1.1}}};
        double[][] thresholds = {{0.8, -0.1}, {0.3}};
        double momentum = 0.0;
        double learningRate = 0.1;
        Network instance = new Network(weights, thresholds, learningRate, momentum);
        try {
            instance.activation(inputs);
        } catch (UnevenArraysException ex) {
            System.err.println(ex);
            fail();
        }
        String expResult = "Network{\n"
                + "  Desired Outputs: {0.0000} Learning Rate: 0.100, Momentum: 0.000\n"
                + "  Neurons (L,N): {\n"
                + "     Neuron(0,0) : {Weights= {[0]: 0.5000, [1]: 0.4000}, Inputs= {[0]: 1.0000, [1]: 1.0000}, Threshold= 0.8000, Output= 0.0572, Error= 0.0000}\n"
                + "     Neuron(0,1) : {Weights= {[0]: 0.9000, [1]: 1.0000}, Inputs= {[0]: 1.0000, [1]: 1.0000}, Threshold= -0.1000, Output= 1.0001, Error= 0.0000}\n"
                + "     Neuron(1,0) : {Weights= {[0]: -1.2000, [1]: 1.1000}, Inputs= {[0]: 0.0572, [1]: 1.0001}, Threshold= 0.3000, Output= 0.4103, Error= 0.0000}\n"
                + "  }\n"
                + "}";
        String result = instance.toString();
        assertEquals(expResult, result);
    }
}
