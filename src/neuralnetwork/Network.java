package neuralnetwork;

import java.util.Random;
import neuralnetworkexceptions.UnevenArraysException;
import java.util.Arrays;
import java.util.Objects;

/*
 Author: Curtis Alcock 18403879
 Project: Artificial Intelligence Algorithm: Back-Propagation with Accelerated Learning
 Title: Neuron
 */
/**
 * A class to control the neural network in its entirety.
 *
 * @author Curtis Alcock 18403879
 */
public class Network {

    /*
     * Array containing all calculating neuron layers.
     */
    private final Neuron[][] mNeurons;

    /*
     * If true then a sigmoidal activation function will be used.
     * If false then a hyperbolic tangent activation function will be used.
     */
    private boolean mActivationFunction;

    /*
     * The desired output of the network.
     */
    private double[] mDesiredOutput;

    /**
     * The learning rate of the network.
     */
    public double mLearningRate;

    /**
     * The momentum constant of the network.
     */
    public double mMomentum;

    public double getMomentum() {
        return mMomentum;
    }

    public void setMomentum(double newMomentum) {
        mMomentum = newMomentum;
    }

    /*
     * The random number generator used in the creation of the neurons.
     */
    private final Random RANDOM;

    /**
     * Initializes the network with weights pseudo-randomly generated according
     * to the given seed.
     *
     * @param neurons the basic structure of the network, i.e. {2,2,1} meaning 2
     * input neurons, 1 hidden layer of 2 neurons and 1 output neuron
     * @param seed the seed to be used when generating the weights.
     * @param learningRate the learning rate constant for the network.
     * @param momentum the momentum constant for the network.
     */
    public Network(int[] neurons, long seed, double learningRate, double momentum) {
        // Initialize the jagged array to have the right number of layers
        mLearningRate = learningRate;
        mMomentum = momentum;
        mActivationFunction = false; // Initially false (meaning will use tanh, which is most efficient), but can be changed in training.
        RANDOM = new Random(seed); // Initialize the generator for genning weights
        mDesiredOutput = new double[neurons[neurons.length - 1]]; // number of neurons in the output layer
        mNeurons = new Neuron[neurons.length - 1][];

        for (int layer = 1; layer < neurons.length; layer++) {
            // Set the number of neurons in each layer
            // so each layer can have a different number of neurons
            mNeurons[layer - 1] = new Neuron[neurons[layer]];

            for (int neuron = 0; neuron < mNeurons[layer - 1].length; neuron++) {
                mNeurons[layer - 1][neuron] = new Neuron(neurons[layer - 1], RANDOM);
            }
        }
    }

    /**
     * Initialize the network to set weight values.
     *
     * @param weights the weights for each link in the network. Each weight is
     * stored at the input side of a Neuron.<br>
     * e.g. {{{W13,W23},{W14,W24}},{W35,W45}}<br>
     * Note: Input neurons do not actually exist, distribution of inputs is
     * handled by the Network (this object).
     * @param thresholds the weight for the bias of each neuron. The bias is to
     * be stored in the same order as the neurons<br>
     * e.g. {{T3,T4},{T5}} for 2 hidden neurons and a single output neuron.
     * @param learningRate the learning rate constant for the network.
     * @param momentum the momentum constant for the network.
     */
    public Network(double[][][] weights, double[][] thresholds, double learningRate, double momentum) {
        mLearningRate = learningRate;
        mMomentum = momentum;
        RANDOM = new Random();

        // Initialize the jagged array to have the right number of layers
        mNeurons = new Neuron[weights.length][];
        for (int layer = 0; layer < weights.length; layer++) {
            // Set the number of neurons in each layer
            mNeurons[layer] = new Neuron[weights[layer].length];
            for (int neuron = 0; neuron < mNeurons[layer].length; neuron++) {
                mNeurons[layer][neuron] = new Neuron(weights[layer][neuron], thresholds[layer][neuron], RANDOM);
            }
        }
        mDesiredOutput = new double[mNeurons[mNeurons.length - 1].length];
    }

    /**
     * Gets the current desired output of the network.
     *
     * @return the current desired output of the network.
     */
    public double[] getDesiredOutput() {
        return mDesiredOutput.clone();
    }

    /**
     * Sets the desired output of the network.
     *
     * @param desiredOutput the output that is desired of the network's
     * activation.
     * @throws UnevenArraysException if the desiredOutput parameter does not
     * match the number of output neurons.
     */
    public void setDesiredOutput(double[] desiredOutput) throws UnevenArraysException {
        if (desiredOutput.length != mDesiredOutput.length) {
            throw new UnevenArraysException("");
        }
        mDesiredOutput = desiredOutput.clone();
    }

    /**
     * Activates the network. Layer by layer, one neuron at a time.
     *
     * @param inputs the input values for the network
     * @return the result of the entire network's activations.
     * @throws UnevenArraysException if the inputs array doesn't match the
     * number of incoming links.
     */
    public double[] activation(double[] inputs) throws UnevenArraysException {
        double[] newInputs;
        newInputs = null;

        // Activate each layer with the outputs of the previous layer
        for (Neuron[] mNeuron : mNeurons) {
            // for each layer
            // Compile an array of the outputs of the current layer
            newInputs = new double[mNeuron.length];
            for (int neuron = 0; neuron < mNeuron.length; neuron++) {
                // for each neuron within a layer
                // add the output to the array
                try {
                    newInputs[neuron] = mNeuron[neuron].activation(inputs, mActivationFunction);
                } catch (UnevenArraysException e) {
                    throw new UnevenArraysException(e.getMessage());
                }
            }
            // Clone the current layer's outputs into the parameter inputs ready for the next layer.
            inputs = newInputs.clone();
        }

        // The result of the final (output) layer's activations
        return newInputs;
    }

    /**
     * Gets a collection of all the weights in the network.
     *
     * @return the weights that determine the output of the network.
     */
    public double[][][] getWeights() {
        double[][][] weights;
        weights = new double[mNeurons.length][][];
        for (int layer = 0; layer < mNeurons.length; layer++) {
            weights[layer] = new double[mNeurons[layer].length][];
            for (int neuron = 0; neuron < mNeurons[layer].length; neuron++) {
                weights[layer][neuron] = mNeurons[layer][neuron].getWeights();
            }
        }
        return weights;
    }

    /**
     * Gets a collection of all the thresholds (biases) in the network.
     *
     * @return all the thresholds in the network.
     */
    public double[][] getThresholds() {
        double[][] thresholds;
        thresholds = new double[mNeurons.length][];
        for (int layer = 0; layer < mNeurons.length; layer++) {
            thresholds[layer] = new double[mNeurons[layer].length];
            for (int neuron = 0; neuron < mNeurons[layer].length; neuron++) {
                thresholds[layer][neuron] = mNeurons[layer][neuron].getThreshold();
            }
        }
        return thresholds;
    }

    /**
     * Back-propagates the error and adjusts the weights for all neurons in the
     * network.
     *
     * @throws UnevenArraysException if the desiredOutput param does not match
     * the number of output neurons.
     */
    public void weightTraining() throws UnevenArraysException {
        double[] gradients;
        double[] weights;
        double[] newGradients;
        int last = mNeurons.length - 1;
        gradients = new double[mNeurons[last].length];
        weights = new double[mNeurons[last].length];

        // for each neuron in the output layer
        for (int neuron = 0; neuron < mNeurons[last].length; neuron++) {
            mNeurons[last][neuron].calcError(mDesiredOutput[neuron]);

            if (mActivationFunction) {
                gradients[neuron] = mNeurons[last][neuron].sigmoidalErrorGradient();
            } else {
                gradients[neuron] = mNeurons[last][neuron].hyperbolicErrorGradient();
            }

            try {
                mNeurons[last][neuron].weightCorrection(mLearningRate, mMomentum, gradients[neuron]);
            } catch (UnevenArraysException ex) {
                throw new UnevenArraysException(String.format("Weight training failed for Neuron(%d,%d): %s", last, neuron, ex));
            }
        }

        // for each hidden layer, count down from the last hidden layer before the outputs
        for (int layer = last - 1; layer >= 0; layer--) {
            newGradients = new double[mNeurons[layer].length];
            for (int neuron = 0; neuron < mNeurons[layer].length; neuron++) {

                for (int prevNeuron = 0; prevNeuron < mNeurons[layer + 1].length; prevNeuron++) {
                    weights[prevNeuron] = mNeurons[layer + 1][prevNeuron].getWeight(neuron);
                }

                if (mActivationFunction) {
                    newGradients[neuron] = mNeurons[layer][neuron].sigmoidalErrorGradient(gradients, weights);
                } else {
                    newGradients[neuron] = mNeurons[layer][neuron].hyperbolicErrorGradient(gradients, weights);
                }

                mNeurons[layer][neuron].weightCorrection(mLearningRate, mMomentum, newGradients[neuron]);
                weights = new double[mNeurons[layer + 1].length];
            }
            gradients = newGradients.clone();
        }

    }

    /**
     * Calculates the Sum of the Squared Errors.
     *
     * @return the sum of the squared errors.
     */
    public double sumOfTheSquaredErrors() {
        double sum;
        int lastLayer;
        sum = 0.0;
        lastLayer = mNeurons.length - 1;

        for (Neuron neuron : mNeurons[lastLayer]) {
            sum += Math.pow(neuron.getError(), 2);
        }   
        return sum;
    }

    /**
     * Runs the training sets through a single pass of the network. This method
     * overload allows for the changing of the Activation Function used in the
     * network process. The TanH function is the most efficient and converges
     * much faster than the Sigmoid function, hence why it is the default.
     *
     * @param trainingSets An array of sets of inputs.
     * @param desiredOutcomes An array of the desired outputs of the network.
     * All second tier arrays should be the same size.
     * @param convergence The point at which the network is to be considered
     * converged.
     * @param function if true then uses sigmoidal activation. If false, uses
     * hyperbolic tangent.
     * @throws UnevenArraysException if the number of inputs in each training
     * set is not equal to the number of input links.
     * @return the sum of the squared errors for this pass.
     */
    public int train(double[][] trainingSets, double[][] desiredOutcomes, double convergence, boolean function) throws UnevenArraysException {
        mActivationFunction = function;
        return this.train(trainingSets, desiredOutcomes, convergence);
    }

    /**
     * Runs the training sets through a single pass of the network.
     *
     * @param trainingSets An array of sets of inputs.
     * @param desiredOutcomes An array of the desired outputs of the network.
     * All second tier arrays should be the same size.
     * @param convergence The point at which the network is to be considered
     * converged.
     * @throws UnevenArraysException if the number of inputs in each training
     * set is not equal to the number of input links.
     * @return the sum of the squared errors for this pass.
     */
    public int train(double[][] trainingSets, double[][] desiredOutcomes, double convergence) throws UnevenArraysException {
        double[] sum = {1, 1};
        int epoch;
        double[][] result;
        String out;
        epoch = 0;
        while (sum[epoch % 2] > convergence) {
            sum[epoch % 2] = 0;
            result = new double[trainingSets.length][];
            for (int i = 0; i < trainingSets.length; i++) {
                try {
                    result[i] = activation(trainingSets[i]);
                    setDesiredOutput(desiredOutcomes[i]);
                    weightTraining();
                    sum[epoch % 2] += sumOfTheSquaredErrors();
                } catch (UnevenArraysException ex) {
                    throw new UnevenArraysException("Training set [" + i + "] was of wrong size: " + ex.getMessage());
                }
            }

            // An attempt at adaptive learning rate; doesn't work as wanted, actually seems to slow down convergence.
            
            if (sum[epoch & 1] / sum[(epoch - 1) & 1] > 1.04) {
                mLearningRate *= 0.7;
                weightTraining();
            } else if (sum[epoch & 1] - sum[(epoch - 1) & 1] < 0.0) {
                mLearningRate *= 1.05;
                weightTraining();
            }
            
            // Epoch report: Outputs the results of each set, the Sum of the Squared Errors, and the Learning Rate.
            System.out.printf("Epoch[%d]:\n", epoch);
            for (int i = 0; i < result.length; i++) {
                out = "{";
                int j;
                for (j = 0; j < result[i].length - 1; j++) {
                    out += String.format("%.4f,", result[i][j]);
                }
                out += String.format("%.4f}", result[i][j]);
                System.out.printf(" Set #%d: %s", i, out); // The output of the network for each activation
            }
            System.out.printf(" SotSE: %.8f", sum[epoch % 2]); // The Sum of the Squared Errors for this epoch.
            System.out.printf(" Learning Rate: %.4f\n", mLearningRate); // The Learning Rate for this epoch.

            epoch++;
        }
        return epoch - 1;
    }

    /**
     * Runs the network forward once over the inputs provided.
     *
     * @param input the inputs to be processed by the network.
     * @return a report of the final values that the network produced.
     * @throws UnevenArraysException if the number of inputs doesn't match the
     * number of input nodes.
     */
    public String run(double[] input) throws UnevenArraysException {
        String out;
        String temp;
        double[] output;
        double[] inputClone;
        double[][][] weights = getWeights();
        double[][] thresholds = getThresholds();
        inputClone = input.clone();
        output = activation(inputClone);
        int i;
        // validate inputs
        if (inputClone.length != weights[0][0].length) {
            throw new UnevenArraysException("There must be exactly "
                    + weights[0][0].length + " input values!");
        }

        // List inputs
        temp = "{";
        for (i = 0; i < input.length - 1; i++) {
            temp += String.format("[%d]=%.4f, ", i, inputClone[i]);
        }
        temp += String.format("[%d]=%.4f}", i, inputClone[i]);
        
        // Insert inputs into final output string
        out = String.format(
                "---------------\n"
                + "| Network Run |\n"
                + "---------------\n"
                + "| Input: %s\n", temp);

        out += "---------------\n"; // section divider
        
        // List weights
        temp = "| Weights: \n";
        for (int layer = 0; layer < weights.length; layer++) {
            // Layer
            temp += String.format("| Layer #%d:\n", layer);
            for (int neuron = 0; neuron < weights[layer].length; neuron++) {
                // Neuron
                temp += String.format("|\tN[%d]: {", neuron);
                int link;
                for (link = 0; link < weights[layer][neuron].length - 1; link++) {
                    // Individual weights
                    temp += String.format("%.4f, ", weights[layer][neuron][link]);
                }
                temp += String.format("%.4f}", weights[layer][neuron][link]);
                temp += String.format(", Threshold: %.4f\n", thresholds[layer][neuron]);
            }
        }
        
        // Add weights to final output
        out += temp;

        out += "---------------\n"; // section divider

        // List outputs
        temp = "";
        for (i = 0; i < output.length - 1; i++) {
            temp += String.format("[%d]=%.4f, ", i, output[i]);
        }
        temp += String.format("[%d]=%.4f", i, output[i]);
        out += String.format("| Output: %s\n", temp);

        return out + "---------------"; // return final output + section divider
    }

    /**
     * Compares two network objects to see if they are equal.
     * Tends to break for networks that utilize randomly generated weights,
     * due to the system time being different when the Random was created,
     * regardless of if the seeds are the same.
     *
     * @param obj2 the network to be compared to.
     * @return true if they are equal.
     */
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
        final Network other = (Network) obj2;
        if (Double.doubleToLongBits(this.mLearningRate) != Double.doubleToLongBits(other.mLearningRate)) {
            return false;
        }
        if (Double.doubleToLongBits(this.mMomentum) != Double.doubleToLongBits(other.mMomentum)) {
            return false;
        }
        boolean same;
        same = true;
        while (same) {
            for (int l = 0; l < this.mNeurons.length; l++) {
                for (int n = 0; n < this.mNeurons[l].length; n++) {
                    same = this.mNeurons[l][n].equals(other.mNeurons[l][n]);
                    if (same == false) {
                        return same;
                    }
                }
            }
        }
        if (!Arrays.equals(this.mDesiredOutput, other.mDesiredOutput)) {
            return false;
        }
        if (!Objects.equals(this.RANDOM, other.RANDOM)) {
            return false;
        }
        return true;
    }

    /**
     * Generates a hashcode unique to this object instance.
     *
     * @return a unique hashcode.
     */
    @Override
    public int hashCode() {
        int hash = 3;
        hash = 97 * hash + Arrays.deepHashCode(this.mNeurons);
        hash = 97 * hash + Arrays.hashCode(this.mDesiredOutput);
        hash = 97 * hash + (int) (Double.doubleToLongBits(this.mLearningRate) ^ (Double.doubleToLongBits(this.mLearningRate) >>> 32));
        hash = 97 * hash + (int) (Double.doubleToLongBits(this.mMomentum) ^ (Double.doubleToLongBits(this.mMomentum) >>> 32));
        hash = 97 * hash + Objects.hashCode(this.RANDOM);
        return hash;
    }

    /**
     * Reports the state of the data members.
     *
     * @return a report of the Network object.
     */
    @Override
    public String toString() {
        String out;
        String outputs;
        int i;
        outputs = "";
        i = 0;
        while (i < mDesiredOutput.length - 1) {
            outputs += String.format("%.4f, ", mDesiredOutput[i]);
            i++;
        }
        outputs += String.format("%.4f", mDesiredOutput[i]);

        out = String.format("Network{\n%20s"
                + "%s} Learning Rate: %.3f, Momentum: %.3f\n%18s\n", "Desired Outputs: {", outputs, mLearningRate, mMomentum, "Neurons (L,N): {");

        for (int layer = 0; layer < mNeurons.length; layer++) {
            for (int neuron = 0; neuron < mNeurons[layer].length; neuron++) {
                out += String.format("%11s(%d,%d) : %s\n", "Neuron", layer, neuron, mNeurons[layer][neuron].toString());
            }
        }

        out += String.format("%3s\n}", "}");

        return out;
    }
}
