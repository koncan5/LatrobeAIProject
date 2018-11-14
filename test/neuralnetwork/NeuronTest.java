/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Random;
import neuralnetworkexceptions.UnevenArraysException;

/**
 * A JUnit test class for the Neuron class.
 *
 * @author 18403879 Curtis Alcock
 */
public class NeuronTest {

    public NeuronTest() {
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
     * Test of getWeight method, of class Neuron.
     */
    @Test
        public void testGetWeight() {
        System.out.println("getWeight");
        int linkId = 0;
        int noInputs = 2;
        Random myRand = new Random(9999);
        Neuron instance = new Neuron(noInputs, myRand);
        double expResult = 0.7629;
        double result = instance.getWeight(linkId);
        assertEquals(expResult, result, 0.0001);
    }

    /**
     * Test of getWeights method, of class Neuron.
     */
    @Test
    public void testGetWeights() {
        System.out.println("getWeights");
        int noInputs = 2;
        Random myRand = new Random(9999);
        Neuron instance = new Neuron(noInputs, myRand);
        double[] expResult = {0.7629, -0.9399};
        double[] result = instance.getWeights();
        assertArrayEquals(expResult, result, 0.0001);
    }

    /**
     * Test of setWeight method, of class Neuron.
     */
    @Test
    public void testSetWeight() {
        System.out.println("setWeight");
        int linkId = 0;
        double newWeight = 0.1984;
        int noInputs = 2;
        Random myRand = new Random(9999);
        Neuron instance = new Neuron(noInputs, myRand);
        instance.setWeight(linkId, newWeight);
        double result = instance.getWeight(linkId);
        assertEquals(newWeight, result, 0.0);
    }

    /**
     * Test of initWeights method, of class Neuron.
     */
    @Test
    public void testInitWeights() {
        System.out.println("initWeights_0args");
        double[] expResult = {0.8241, -0.3902};
        int noInputs = 2;
        Random myRand = new Random(9999);
        Neuron instance = new Neuron(noInputs, myRand);
        instance.initWeights();
        double[] result = instance.getWeights();
        assertArrayEquals(expResult, result, 0.0001);
    }

    /**
     * Test of setWeights method, of class Neuron.
     */
    @Test
    public void testSetWeights() {
        System.out.println("setWeights");
        double[] newWeights = {0.1984, -0.6581};
        int noInputs = 2;
        Random myRand = new Random(9999);
        Neuron instance = new Neuron(noInputs, myRand);
        instance.setWeights(newWeights);
        double[] result = instance.getWeights();
        assertArrayEquals(newWeights, result, 0.0);
    }

    /**
     * Test of getOutput method, of class Neuron.
     */
    @Test
    public void testGetOutput() {
        System.out.println("getOutput");
        double[] weights = {0.5, 0.4};
        double threshold = 0.8;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double[] inputs = {1, 0};
        double expResult = 0.4256;
        try {
            instance.activation(inputs, true);
        } catch (UnevenArraysException e) {
            System.err.printf("Activation failed: ", e);
            fail();
        }
        double result = instance.getOutput();
        assertEquals(expResult, result, 0.0001);
    }

    /**
     * Test of getThreshold method, of class Neuron.
     */
    @Test
    public void testGetThreshold() {
        System.out.println("getThreshold");
        double[] weights = {0.5, 0.4};
        double threshold = 0.8;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double expResult = 0.8;
        double result = instance.getThreshold();
        assertEquals(expResult, result, 0.0);
    }

    /**
     * Test of initThreshold method, of class Neuron.
     */
    @Test
    public void testInitThreshold() {
        System.out.println("initThreshold");
        double expResult = 0.8241;
        int noInputs = 2;
        Random myRand = new Random(9999);
        Neuron instance = new Neuron(noInputs, myRand);
        instance.initThreshold();
        double result = instance.getThreshold();
        assertEquals(expResult, result, 0.0001);

    }

    /**
     * Test of setThreshold method, of class Neuron.
     */
    @Test
    public void testSetThreshold() {
        System.out.println("setThreshold");
        double newThreshold = 0.0231;
        int noInputs = 2;
        Random myRand = new Random(9999);
        Neuron instance = new Neuron(noInputs, myRand);
        instance.setThreshold(newThreshold);
        double result = instance.getThreshold();
        assertEquals(newThreshold, result, 0.0);
    }

    /**
     * Test of getError method, of class Neuron.
     */
    @Test
    public void testGetError() {
        System.out.println("getError");
        double[] inputs = {0, 1};
        double[] weights = {-1.2, 1.1};
        double threshold = 0.3;
        double desiredOutput = 0.9;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        try {
            instance.activation(inputs, false);
        } catch (UnevenArraysException ex) {
            fail(ex.toString());
        }
        instance.calcError(desiredOutput);
        double expResult = 0.4529;
        double result = instance.getError();
        assertEquals(expResult, result, 0.0001);
    }

    /**
     * Test of activation method, of class Neuron, using the Sigmoid activation
     * function.
     */
    @Test
    public void testActivation_Sigmoidal() {
        System.out.println("activation_Sigmoidal");
        double[] inputs = {1, 1};
        double[] weights = {0.5, 0.4};
        double threshold = 0.8;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double expResult = 0.5250;
        double result = 0.0;
        try {
            result = instance.activation(inputs, true);
        } catch (UnevenArraysException e) {
            System.err.printf("Activation failed: ", e);
            fail();
        }
        assertEquals(expResult, result, 0.0001);
    }

    /**
     * Test of activation method, of class Neuron, using the TanH activation
     * function.
     */
    @Test
    public void testActivation_TanH() {
        System.out.println("activation_TanH");
        double[] inputs = {1, 1};
        double[] weights = {0.5, 0.4};
        double threshold = 0.8;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double expResult = 0.0572;
        double result = 0.0;
        try {
            result = instance.activation(inputs, false);
        } catch (UnevenArraysException e) {
            System.err.printf("Activation failed: ", e);
            fail();
        }
        assertEquals(expResult, result, 0.0001);
    }

    /**
     * Test of setError method, of class Neuron.
     */
    @Test
    public void testCalcError() {
        System.out.println("calcError");
        double expResult = -0.5097;
        double desiredOutput = 0;
        double[] weights = {-1.2, 1.1};
        double threshold = 0.3;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double[] inputs = {0.5250, 0.8808};
        try {
            instance.activation(inputs, true);
        } catch (UnevenArraysException e) {
            System.err.printf("Activation failed: ", e);
            fail();
        }
        instance.calcError(desiredOutput);
        double result = instance.getError();
        assertEquals(expResult, result, 0.0001);
    }

    /**
     * Test of toString method, of class Neuron.
     */
    @Test
    public void testToString() {
        System.out.println("toString");
        double desiredOutput = 0;
        double[] weights = {-1.2, 1.1};
        double threshold = 0.3;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double[] inputs = {0.5250, 0.8808};
        try {
            instance.activation(inputs, true);
        } catch (UnevenArraysException e) {
            System.err.printf("Activation failed: ", e);
            fail();
        }
        instance.calcError(desiredOutput);
        String expResult = "{Weights= {[0]: -1.2000, [1]: 1.1000}, Inputs= {[0]: 0.5250, [1]: 0.8808}, Threshold= 0.3000, Output= 0.5097, Error= -0.5097}";
        String result = instance.toString();
        assertEquals(expResult, result);
    }

    /**
     * Test of getInputs method, of class Neuron.
     */
    @Test
    public void testGetInputs() {
        System.out.println("getInputs");
        double[] weights = {-1.2, 1.1};
        double threshold = 0.3;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double[] inputs = {0.5250, 0.8808};
        try {
            instance.activation(inputs, true);
        } catch (UnevenArraysException e) {
            System.err.printf("Activation failed: ", e);
            fail();
        }
        double[] result = instance.getInputs();
        assertArrayEquals(inputs, result, 0.0);
    }

    /**
     * Test of sigmoidalErrorGradient method, of class Neuron.
     */
    @Test
    public void testSigmoidalErrorGradient_0args() {
        System.out.println("sigmoidalErrorGradient_0args");
        double desiredOutput = 0;
        double[] weights = {-1.2, 1.1};
        double threshold = 0.3;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double[] inputs = {0.5250, 0.8808};
        try {
            instance.activation(inputs, true);
        } catch (UnevenArraysException e) {
            System.err.printf("Activation failed: ", e);
            fail();
        }
        instance.calcError(desiredOutput);
        double expResult = -0.1274;
        double result = instance.sigmoidalErrorGradient();
        assertEquals(expResult, result, 0.0001);
    }

    /**
     * Test of sigmoidalErrorGradient method, of class Neuron.
     */
    @Test
    public void testSigmoidalErrorGradient_doubleArr_doubleArr() {
        System.out.println("sigmoidalErrorGradient_doubleArr_doubleArr");
        double[] weights = {0.5, 0.4};
        double threshold = 0.8;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double[] inputs = {1, 0};
        double[] gradients = {0.1254};
        double[] backWeights = {-1.2};
        double expResult = -0.0368;
        double result = 0.0;
        try {
            instance.activation(inputs, true);
            result = instance.sigmoidalErrorGradient(gradients, backWeights);
        } catch (UnevenArraysException e) {
            System.err.printf("Error Gradient calculation failed: ", e);
            fail();
        }
        assertEquals(expResult, result, 0.0001);
    }

    /**
     * Test of hyperbolicErrorGradient method, of class Neuron.
     */
    @Test
    public void testHyperbolicErrorGradient_0args() {
        System.out.println("hyperbolicErrorGradient_0args");
        double desiredOutput = 0;
        double[] weights = {-1.2, 1.1};
        double threshold = 0.3;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double[] inputs = {0.5250, 0.8808};
        try {
            instance.activation(inputs, false);
        } catch (UnevenArraysException e) {
            System.err.printf("Activation failed: ", e);
            fail();
        }
        instance.calcError(desiredOutput);
        double expResult = -0.0127;
        double result = instance.hyperbolicErrorGradient();
        assertEquals(expResult, result, 0.0001);
    }

    /**
     * Test of hyperbolicErrorGradient method, of class Neuron.
     */
    @Test
    public void testHyperbolicErrorGradient_doubleArr_doubleArr() {
        System.out.println("hyperbolicErrorGradient_doubleArr_doubleArr");
        double[] weights = {0.5, 0.4};
        double threshold = 0.8;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double[] inputs = {1, 0};
        double[] gradients = {0.1254};
        double[] backweights = {-1.2};
        double expResult = -0.0852;
        double result = 0.0;
        try {
            instance.activation(inputs, false);
            result = instance.hyperbolicErrorGradient(gradients, backweights);
        } catch (UnevenArraysException e) {
            System.err.printf("Error Gradient calculation failed: ", e);
            fail();
        }
        assertEquals(expResult, result, 0.0001);
    }

    /**
     * Test of weightCorrection method, of class Neuron.
     */
    @Test
    public void testWeightCorrection() {
        System.out.println("weightCorrection");
        double[] weights = {0.5, 0.4};
        double threshold = 0.8;
        Random myRand = new Random();
        Neuron instance = new Neuron(weights, threshold, myRand);
        double[] inputs = {1, 0};
        double[] gradients = {0.1254};
        double[] backweights = {-1.2};
        double desiredOutput = 0.9;
        double gradient;
        double[] expResult = {0.4963, 0.4};
        double[] result;
        double learningRate = 0.1;
        double momentum = 0;
        try {
            instance.activation(inputs, true);
            instance.calcError(desiredOutput);
            gradient = instance.sigmoidalErrorGradient(gradients, backweights);
            instance.weightCorrection(learningRate, momentum, gradient);
        } catch (UnevenArraysException e) {
            System.err.println(e);
            fail();
        }
        result = instance.getWeights();
        assertArrayEquals(expResult, result, 0.0001);
    }

}
