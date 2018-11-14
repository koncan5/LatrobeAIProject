package neuralnetwork;

/*
 Author: Curtis Alcock 18403879
 Project: Artificial Intelligence Algorithm: Back-Propagation with Accelerated Learning
 Title: Neuron
 */
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import neuralnetworkexceptions.UnevenArraysException;

/**
 * Defines the neurons in the network.
 *
 * @author Curtis Alcock 18403879
 */
public class Neuron {

    /*
     * An array of doubles containing the weights of each link to the next
     * higher layer.
     */
    private double[] mWeights;

    /*
     * The bias weight of the neuron.
     */
    private double mThreshold;

    /*
     * The inputs that were passed to the neuron in the most recent activation.
     */
    private double[] mInputs;

    /*
     * The output of the neuron for this iteration.
     */
    private double mOutput;

    /*
     * The error of the neuron for this iteration.
     */
    private double mError;

    /*
     * The previous delta for each link, used in calculating the next one.
     */
    private final double[] mPreviousWeightsDelta;

    /*
     * The previous delta for the threshold, used in calculating the next one.
     */
    private double mPreviousThresholdDelta;

    /*
     * A random number generator to be passed from Network on initialization.
     */
    private final Random RANDOM;

    /*
     * A scalar constant for the hyperbolic tangent activation function,
     * enables accelerated learning of the network.
     * 
     * Represents 'a' in Eq6.16 (Guyon, 1991; Negnevistky, 2011).
     */
    public static final double HYPERBOLIC_TANGENT_A = 1.716;

    /*
     * A scalar constant for the hyperbolic tangent activation function,
     * enables accelerated learning of the network.
     * 
     * Represents 'b' in Eq6.16 (Guyon, 1991; Negnevistky, 2011).
     */
    public static final double HYPERBOLIC_TANGENT_B = 2.0 / 3.0;

    /**
     * Initializes the Neuron with a set number of inputs. All neurons have an
     * initial output of 0.
     *
     * @param noInputs the number of input links this neuron has.
     * @param myRand random number generator to generate the weights.
     */
    public Neuron(int noInputs, Random myRand) {
        mWeights = new double[noInputs]; // set the number of links to account for
        mInputs = new double[noInputs]; // all inputs are 0 prior to activation
        mOutput = 0.0; // the initial output of all neurons is 0
        mError = 0.0; // error should always initialize at 0
        mPreviousWeightsDelta = new double[noInputs]; // previous delta is initialized at 0
        mPreviousThresholdDelta = 0; // previous delta is initially 0 
        RANDOM = myRand; // So that all Neurons are using the same RNG

        initWeights(); // randomly assign weights
        initThreshold();
    }

    /**
     * Instantiates the Neuron with predefined weights and threshold. The RNG is
     * in case the weights are randomly reinitialized.
     *
     * @param weights
     * @param threshold
     * @param myRand
     */
    public Neuron(double[] weights, double threshold, Random myRand) {
        mWeights = weights.clone();
        mInputs = new double[weights.length];
        mOutput = 0.0; // the initial output of all neurons is 0
        mThreshold = threshold;
        mError = 0.0; // error should always initialize at 0
        mPreviousWeightsDelta = new double[weights.length];
        mPreviousThresholdDelta = 0;
        RANDOM = myRand;
    }

    /**
     * Gets the value of the weight for a specified link.
     *
     * @param linkId the link whose value we are getting.
     * @return the weight of the link.
     */
    public double getWeight(int linkId) {
        return mWeights[linkId];
    }

    /**
     * Gets a copy of the weights array for this neuron.
     *
     * @return the array of weights in the same order as the neurons they
     * pertain to.
     */
    public double[] getWeights() {
        return mWeights.clone();
    }

    /**
     * Sets an individual links weight to the specified value.
     *
     * @param linkId the link whose weight is to be modified.
     * @param newWeight the new value for the links weight.
     */
    public void setWeight(int linkId, double newWeight) {
        mWeights[linkId] = newWeight;
    }

    /**
     * Initializes the weights to random values between -1 to +1.
     */
    public final void initWeights() {
        for (int i = 0; i < mWeights.length; i++) {
            mWeights[i] = RANDOM.nextDouble() * 2 - 1;
        }
    }

    /**
     * Sets all the weights to the specified array.
     *
     * @param newWeights the array of weights to be applied to the input links.
     */
    public void setWeights(double[] newWeights) {
        mWeights = newWeights.clone();
    }

    /**
     * Gets the most recent set of inputs for this neuron.
     *
     * @return the most recent set of inputs for this neuron.
     */
    public double[] getInputs() {
        return mInputs.clone();
    }

    /**
     * Gets the most recent output of this neuron.
     *
     * @return the value of the last output.
     */
    public double getOutput() {
        return mOutput;
    }

    /**
     * Gets the Threshold of the Neuron. Also known as the bias.
     *
     * @return the Threshold of the Neuron
     */
    public double getThreshold() {
        return mThreshold;
    }

    /**
     * Initializes the Threshold to a random value between -1 and +1.
     */
    public final void initThreshold() {
        mThreshold = RANDOM.nextDouble() * 2 - 1;
    }

    /**
     * Sets the Threshold to the specified value.
     *
     * @param newThreshold the specified value to be set.
     */
    public void setThreshold(double newThreshold) {
        mThreshold = newThreshold;
    }

    /**
     * Gets the current Error of the neuron. In hidden layers this will be the
     * error gradient.
     *
     * @return the current error.
     */
    public double getError() {
        return mError;
    }

    /**
     * Weights the inputs and sums them together.
     *
     * @return the sum of the weighted inputs.
     */
    private double weightedSum() {
        double output;
        output = 0;

        try {
            for (int i = 0; i < mInputs.length; i++) {
                output += mInputs[i] * mWeights[i]; // summing the weighted inputs
            }
        } catch (ArrayIndexOutOfBoundsException | NullPointerException e) {
            System.err.printf("Activation failed, different array lengths for inputs and weights:\n%s", e);
            System.err.printf("Weighted summing stopped at: %.4f", output);
        }

        return output;
    }

    /**
     * Sigmoid Transfer Function. Converts any number from between (-infinity,
     * +infinity) to a value between (0, 1).
     *
     * @param input a value to be transfered to the binary space.
     * @return a relative number between 0 and +1
     */
    private double sigmoidTransfer(double input) {
        return 1 / (1 + Math.exp(-1.0 * (input - mThreshold))); // sigmoid activation function
    }

    /**
     * Hyperbolic Tangent Transfer Function. Converts any number from between
     * (-infinity, +infinity) to a value between (-1, 1).
     *
     * @param input a value to be transfered to the binary space.
     * @return a relative number between -1 and +1
     */
    private double hyperbolicTransfer(double input) {
        return ((2.0 * HYPERBOLIC_TANGENT_A) / (1.0 + Math.exp(-HYPERBOLIC_TANGENT_B * (input - mThreshold)))) - HYPERBOLIC_TANGENT_A; // Hyperbolic tangent activation function
    }

    /**
     * Activates the neuron. For the network to work correctly, you must also
     * use the error gradient function that corresponds to the activation
     * function that you are using.
     *
     * @param inputs the input values in the same order as the neurons in the
     * previous layer. Must have the same number of values as there are weights!
     * @param function if true then uses sigmoidal activation. If false, uses
     * hyperbolic tangent.
     * @return the output value of this neuron.
     * @throws UnevenArraysException if there are a different number of inputs
     * to weights.
     */
    public double activation(double[] inputs, boolean function) throws UnevenArraysException {
        // A little validation
        if (inputs.length != mWeights.length) {
            throw new UnevenArraysException();
        }

        mInputs = inputs.clone();
        if (function) {
            mOutput = sigmoidTransfer(weightedSum());
        } else {
            mOutput = hyperbolicTransfer(weightedSum());
        }
        return mOutput;
    }

    /**
     * Calculates the error of the network. Compares the desired output to that
     * which was produced by the network.
     *
     * @param desiredOutput the desired output of the network
     */
    public void calcError(double desiredOutput) {
        mError = desiredOutput - mOutput;
    }

    /**
     * Method for calculating the error gradient for Output neurons when using
     * the Sigmoidal activation function.
     *
     * @return the error gradient for this neuron.
     */
    public double sigmoidalErrorGradient() {
        double gradient;
        gradient = mOutput * (1 - mOutput) * mError;
        return gradient;
    }

    /**
     * Calculates the Error Gradient for hidden Neurons when using the Sigmoidal
     * activation function.
     *
     * @param gradients the gradients for all neurons in the layer below (next)
     * @param weights the weights for all outgoing links
     * @return the error gradient for this neuron.
     * @throws UnevenArraysException if the gradients and weights parameters are
     * of differing lengths.
     */
    public double sigmoidalErrorGradient(double[] gradients, double[] weights) throws UnevenArraysException {
        double gradient;
        gradient = 0.0;

        if (gradients.length != weights.length) {
            throw new UnevenArraysException("The gradients and weights arrays are of differing lengths!");
        }

        // summing the weighted gradients
        // each gradient should have a corresponding weight assosciated
        for (int i = 0; i < gradients.length; i++) {
            gradient += gradients[i] * weights[i];
        }

        gradient = mOutput * (1 - mOutput) * gradient;
        return gradient;
    }

    /**
     * Method for calculating the error gradient for Output neurons when using
     * the Hyperbolic Tangent activation function.
     *
     * @return the error gradient for this neuron.
     */
    public double hyperbolicErrorGradient() {
        double gradient;
        double sum;
        sum = weightedSum() - mThreshold;
        gradient = ((2 * HYPERBOLIC_TANGENT_A * HYPERBOLIC_TANGENT_B * Math.exp(HYPERBOLIC_TANGENT_B * sum)) / Math.pow(Math.exp(HYPERBOLIC_TANGENT_B * sum) + 1.0, 2)) * mError;
        return gradient;
    }

    /**
     * Calculates the Error Gradient for hidden Neurons when using the
     * Hyperbolic Tangent activation function.
     *
     * @param gradients the gradients for all neurons in the layer below (next)
     * @param weights the weights for all outgoing links
     * @return the error gradient for this neuron.
     * @throws UnevenArraysException if the gradients and weights parameters are
     * of differing lengths.
     */
    public double hyperbolicErrorGradient(double[] gradients, double[] weights) throws UnevenArraysException {
        double gradient;
        double sum;
        gradient = 0.0;
        sum = weightedSum() - mThreshold;

        if (gradients.length != weights.length) {
            throw new UnevenArraysException("The gradients and weights arrays are of differing lengths!");
        }

        // summing the weighted gradients
        // each gradient should have a corresponding weight assosciated
        for (int i = 0; i < gradients.length; i++) {
            gradient += gradients[i] * weights[i];
        }

        gradient = ((2 * HYPERBOLIC_TANGENT_A * HYPERBOLIC_TANGENT_B * Math.exp(HYPERBOLIC_TANGENT_B * sum)) / Math.pow(Math.exp(HYPERBOLIC_TANGENT_B * sum) + 1.0, 2)) * gradient;
        return gradient;
    }

    /**
     * Updates the weights for this neuron.
     *
     * @param learningRate the learning rate parameter.
     * @param momentum the current momentum of the learning algorithm.
     * @param gradient the error gradient for this neuron for this iteration.
     * @throws UnevenArraysException If for some reason the weights and inputs
     * member arrays are of differing lengths.
     */
    public void weightCorrection(double learningRate, double momentum, double gradient) throws UnevenArraysException {
        double delta;

        // A little validation
        if (mWeights.length != mInputs.length) {
            throw new UnevenArraysException("The weights and inputs arrays are of differing lengths!");
        }

        // calculate the weight correction delta for each link
        for (int link = 0; link < mWeights.length; link++) {
            delta = momentum * mPreviousWeightsDelta[link] + learningRate * mInputs[link] * gradient;

            mWeights[link] += delta; // update the link's weight
            mPreviousWeightsDelta[link] = delta; // update the delta for next iteration
        }

        // do the same for the threshold
        delta = momentum * mPreviousThresholdDelta + learningRate * -1 * gradient;
        mThreshold += delta;
        mPreviousThresholdDelta = delta;
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 97 * hash + Arrays.hashCode(this.mWeights);
        hash = 97 * hash + (int) (Double.doubleToLongBits(this.mThreshold) ^ (Double.doubleToLongBits(this.mThreshold) >>> 32));
        hash = 97 * hash + Arrays.hashCode(this.mInputs);
        hash = 97 * hash + (int) (Double.doubleToLongBits(this.mOutput) ^ (Double.doubleToLongBits(this.mOutput) >>> 32));
        hash = 97 * hash + (int) (Double.doubleToLongBits(this.mError) ^ (Double.doubleToLongBits(this.mError) >>> 32));
        hash = 97 * hash + Objects.hashCode(this.RANDOM);
        return hash;
    }

    @Override
    public boolean equals(Object obj2) {
        if (this == obj2) {
            return true;
        }
        if (obj2 == null) {
            return false;
        }
        if (getClass() != obj2.getClass()) {
            return false;
        }
        final Neuron other = (Neuron) obj2;
        if (Double.doubleToLongBits(this.mThreshold) != Double.doubleToLongBits(other.mThreshold)) {
            System.out.println("Threshold");
            return false;
        }
        if (Double.doubleToLongBits(this.mOutput) != Double.doubleToLongBits(other.mOutput)) {
            System.out.println("Output");
            return false;
        }
        if (Double.doubleToLongBits(this.mError) != Double.doubleToLongBits(other.mError)) {
            System.out.println("Error");
            return false;
        }

        if (!Arrays.equals(this.mInputs, other.mInputs)) {
            System.out.println("Inputs");
            return false;
        }
        if (!Objects.equals(this.RANDOM, other.RANDOM)) {
            System.out.println("Random");
            return false;
        }
        if (this.mWeights.length == other.mWeights.length) {
            boolean same;
            int i = 0;
            while (same = true || i < this.mWeights.length) {
                if (Math.abs(((Double) this.mWeights[i]).compareTo(other.mWeights[i])) < 0.0001) {
                    System.out.println(Math.abs(((Double) this.mWeights[i]).compareTo(other.mWeights[i])));
                    same = false;
                }
                i++;
            }
            System.out.println("Arrays same");
            return same;
        } else {
            System.out.println("Arrays different");
            return false;
        }
    }

    /**
     * Outputs the state of the Neuron as a string.
     *
     * @return the state of the neuron.
     */
    @Override
    public String toString() {
        String weights;
        String inputs;
        weights = "";
        inputs = "";

        int i = 0;
        while (i < mWeights.length - 1) {
            weights += String.format("[%d]: %.4f, ", i, mWeights[i]);
            i++;
        }
        weights += String.format("[%d]: %.4f", i, mWeights[i]);

        i = 0;
        while (i < mInputs.length - 1) {
            inputs += String.format("[%d]: %.4f, ", i, mInputs[i]);
            i++;
        }
        inputs += String.format("[%d]: %.4f", i, mInputs[i]);

        return String.format("{Weights= {%s}, Inputs= {%s}, Threshold= %.4f, Output= %.4f, Error= %.4f}", weights, inputs, mThreshold, mOutput, mError);
    }
}
