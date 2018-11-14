package io;

import java.io.*;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.w3c.dom.*;
import javax.xml.parsers.*;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import neuralnetwork.Network;
import org.xml.sax.SAXException;

/*
 Author: Curtis Alcock 18403879
 Lab: Assessment1 
 Title: IOManager
 */
/**
 * The IOManager class handles file IO for my Back-Prop Network.
 *
 * @author Curtis Alcock 18403879
 */
public class IOManager {

    /**
     * Instantiates a Back-Prop Network from an XML file.
     *
     * <ul>
     * <li>
     * Network that pseudo-randomly generates weights with a seed. DOM Structure
     * Example:
     * <p>
     * &lt;project&gt;<br>
     * &nbsp;&lt;network&gt;<br>
     * &nbsp;&nbsp;&lt;learningRate&gt;0.1&lt;/learningRate&gt;<br>
     * &nbsp;&nbsp;&lt;momentum&gt;0&lt;/momentum&gt;<br>
     * &nbsp;&nbsp;&lt;seed&gt;6969&lt;/seed&gt;<br>
     * &nbsp;&nbsp;&lt;neurons&gt;2,2,1&lt;/neurons&gt;<br>
     * &nbsp;&lt;/network&gt;<br>
     * &lt;/project&gt;
     * </p>
     * </li>
     * <li>
     * Network that is fully fleshed out with set values, guaranteed same
     * results every time. DOM Structure Example:
     * <p>
     * &lt;project&gt;<br>
     * &nbsp;&lt;network&gt;<br>
     * &nbsp;&nbsp;&lt;learningRate&gt;0.1&lt;/learningRate&gt;<br>
     * &nbsp;&nbsp;&lt;momentum&gt;0&lt;/momentum&gt;<br>
     * &nbsp;&nbsp;&lt;neurons&gt;<br>
     * &nbsp;&nbsp;&nbsp;&lt;layer&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&lt;neuron&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;weight&gt;0.5&lt;/weight&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;weight&gt;0.4&lt;/weight&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;threshold&gt;0.8&lt;/threshold&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&lt;/neuron&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&lt;neuron&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;weight&gt;0.9&lt;/weight&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;weight&gt;1.0&lt;/weight&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;threshold&gt;-0.1&lt;/threshold&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&lt;/neuron&gt;<br>
     * &nbsp;&nbsp;&nbsp;&lt;/layer&gt;<br>
     * &nbsp;&nbsp;&nbsp;&lt;layer&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&lt;neuron&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;weight&gt;-1.2&lt;/weight&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;weight&gt;1.1&lt;/weight&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;threshold&gt;0.3&lt;/threshold&gt;<br>
     * &nbsp;&nbsp;&nbsp;&nbsp;&lt;/neuron&gt;<br>
     * &nbsp;&nbsp;&nbsp;&lt;/layer&gt;<br>
     * &nbsp;&nbsp;&lt;/neurons&gt;<br>
     * &nbsp;&lt;/network&gt;<br>
     * &lt;/project&gt;
     * </p>
     * </li>
     * </ul>
     *
     * @param file the XML file to be read.
     * @return a Back-Prop Network, fully initialized with the values from file.
     * Null if file not found.
     */
    public static Network readNetwork(File file) {
        Network n;
        double learningRate;
        double momentum;

        try {
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(file);
            doc.getDocumentElement().normalize();

            System.out.println("Reading from xml file: " + file.getPath() + " " + file.getName());

            Element eElement = (Element) doc.getElementsByTagName("network").item(0);
            learningRate = Double.parseDouble(eElement.getElementsByTagName("learningRate")
                    .item(0).getTextContent());
            momentum = Double.parseDouble(eElement.getElementsByTagName("momentum")
                    .item(0).getTextContent());
            Element neurons = (Element) eElement.getElementsByTagName("neurons")
                    .item(0);

            if (neurons.getElementsByTagName("layer").getLength() > 0) {
                double[][][] weights;
                double thresholds[][];
                weights = new double[neurons
                        .getElementsByTagName("layer")
                        .getLength()][][];
                thresholds = new double[neurons
                        .getElementsByTagName("layer")
                        .getLength()][];

                // Iterate over each layer
                for (int l = 0; l < weights.length; l++) {
                    weights[l] = new double[((Element) neurons.getElementsByTagName("layer") // Get the list of layers
                            .item(l)) // Get layer 'l'
                            .getElementsByTagName("neuron") // Get the list of neurons in layer 'l'
                            .getLength()][]; // Get the length of that list
                    thresholds[l] = new double[((Element) neurons.getElementsByTagName("layer")
                            .item(l)).getElementsByTagName("neuron")
                            .getLength()];

                    // Iterate over each neuron
                    for (int neu = 0; neu < weights[l].length; neu++) {
                        weights[l][neu] = new double[((Element) ((Element) neurons
                                .getElementsByTagName("layer")
                                .item(l))
                                .getElementsByTagName("neuron")
                                .item(neu))
                                .getElementsByTagName("weight")
                                .getLength()];

                        // Iterate over the weights
                        for (int link = 0; link < weights[l][neu].length; link++) {
                            // Parse the weight
                            weights[l][neu][link] = Double.parseDouble(((Element) ((Element) neurons
                                    .getElementsByTagName("layer")
                                    .item(l))
                                    .getElementsByTagName("neuron")
                                    .item(neu))
                                    .getElementsByTagName("weight")
                                    .item(link)
                                    .getTextContent());
                        }
                        // Parse the threshold
                        thresholds[l][neu] = Double.parseDouble(((Element) ((Element) neurons
                                .getElementsByTagName("layer")
                                .item(l))
                                .getElementsByTagName("neuron")
                                .item(neu))
                                .getElementsByTagName("threshold")
                                .item(0)
                                .getTextContent());
                    }
                }

                // Instantiate the Network object
                n = new Network(weights, thresholds, learningRate, momentum);

            } else {
                int[] noNeurons;
                long seed;
                noNeurons = stringToIntArray(eElement.getElementsByTagName("neurons")
                        .item(0)
                        .getTextContent(), ",");
                seed = Long.parseLong(eElement.getElementsByTagName("seed")
                        .item(0)
                        .getTextContent());

                // Instantiate the Network object
                n = new Network(noNeurons, seed, learningRate, momentum);
            }

            return n;

        } catch (FileNotFoundException f) {
            Logger.getLogger(IOManager.class.getName()).log(Level.SEVERE, "File Not found, no network...", f);
        } catch (ParserConfigurationException | SAXException | IOException | DOMException e) {
            Logger.getLogger(IOManager.class.getName()).log(Level.SEVERE, null, e);
        }

        return null;
    }

    /**
     * Reads the sets of inputs to be used when training a neural network from
     * an XML file.
     *
     * DOM Structure Example:
     * <p>
     * &gt;project&lt;<br>
     * &nbsp;&gt;trainingSets&lt;<br>
     * &nbsp;&nbsp;&gt;set&lt;0,0&gt;/set&lt;<br>
     * &nbsp;&nbsp;&gt;set&lt;0,1&gt;/set&lt;<br>
     * &nbsp;&nbsp;&gt;set&lt;1,1&gt;/set&lt;<br>
     * &nbsp;&nbsp;&gt;set&lt;1,0&gt;/set&lt;<br>
     * &nbsp;&gt;/trainingSets&lt;<br>
     * &gt;/project&lt;
     * </p>
     *
     * @param file the XML file to be read.
     * @return sets of inputs for use when training a network.
     */
    public static double[][] readTrainingSets(File file) {
        double[][] trainingSets;

        try {
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(file);
            doc.getDocumentElement().normalize();

            Element sets = (Element) doc.getElementsByTagName("trainingSets").item(0);
            trainingSets = new double[sets.getElementsByTagName("set").getLength()][];

            for (int iter = 0; iter < trainingSets.length; iter++) {
                trainingSets[iter] = stringToDoubleArray(sets
                        .getElementsByTagName("set")
                        .item(iter)
                        .getTextContent(), ",");
            }

            return trainingSets;
        } catch (FileNotFoundException f) {
            Logger.getLogger(IOManager.class.getName()).log(Level.SEVERE, "File not found...", f);
        } catch (ParserConfigurationException | SAXException | IOException | DOMException e) {
            Logger.getLogger(IOManager.class.getName()).log(Level.SEVERE, "Training Sets reading failed...", e);
        }
        return null;
    }

    /**
     * Reads the Desired Outcomes from an XML file.
     * <p>
     * &lt;desiredOutcomes&gt;<br>
     * &nbsp;&lt;set&gt;0&lt;/set&gt;<br>
     * &nbsp;&lt;set&gt;1&lt;/set&gt;<br>
     * &nbsp;&lt;set&gt;0&lt;/set&gt;<br>
     * &nbsp;&lt;set&gt;1&lt;/set&gt;<br>
     * &lt;/desiredOutcomes&gt;
     * </p>
     *
     * @param file the XML file to be read.
     * @return an array of arrays of doubles containing the desired outputs for
     * each training set.
     */
    public static double[][] readDesiredOutcomes(File file) {
        double[][] desiredOutcomes;

        try {
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(file);
            doc.getDocumentElement().normalize();

            Element sets = (Element) doc.getElementsByTagName("desiredOutcomes").item(0);
            desiredOutcomes = new double[sets.getElementsByTagName("set").getLength()][];

            for (int iter = 0; iter < desiredOutcomes.length; iter++) {
                desiredOutcomes[iter] = stringToDoubleArray(sets
                        .getElementsByTagName("set")
                        .item(iter)
                        .getTextContent(), ",");
            }

            return desiredOutcomes;
        } catch (FileNotFoundException f) {
            Logger.getLogger(IOManager.class.getName()).log(Level.SEVERE, "File not found...", f);
        } catch (ParserConfigurationException | SAXException | IOException | DOMException e) {
            Logger.getLogger(IOManager.class.getName()).log(Level.SEVERE, "Training Sets reading failed...", e);
        }

        return null;
    }

    /**
     * Writes the Network to an XML file in the project's root directory. Same
     * DOM Structure as a fully set network. See {@link #readNetwork}.
     *
     * @see readNetwork
     * @param file the location and name of the file to be written.
     * @param n the database to be saved.
     */
    public static void writeNetwork(File file, Network n) {
        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder;
        try {
            // Build the structure and assign the DVD details to the structure
            dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.newDocument();
            Element rootElement = doc.createElement("collection");
            doc.appendChild(rootElement);

            Element network = doc.createElement("network");
            rootElement.appendChild(network);

            Element neurons = doc.createElement("neurons");
            network.appendChild(neurons);
            double[][][] weights = n.getWeights();
            double[][] thresholds = n.getThresholds();
            for (int l = 0; l < weights.length; l++) {
                Element layer = doc.createElement("layer");
                neurons.appendChild(layer);
                for (int neu = 0; neu < weights[l].length; neu++) {
                    Element neuron = doc.createElement("neuron");
                    layer.appendChild(neuron);
                    for (int w = 0; w < weights[l][neu].length; w++) {
                        Element weight = doc.createElement("weight");
                        weight.setTextContent(Double.toString(weights[l][neu][w]));
                        neuron.appendChild(weight);
                    }

                    Element threshold = doc.createElement("threshold");
                    threshold.setTextContent(Double.toString(thresholds[l][neu]));
                    neuron.appendChild(threshold);
                }
            }

            Element learningRate = doc.createElement("learningRate");
            learningRate.setTextContent(Double.toString(n.mLearningRate));
            network.appendChild(learningRate);

            Element momentum = doc.createElement("momentum");
            momentum.setTextContent(Double.toString(n.mMomentum));
            network.appendChild(momentum);

            // Write the XML
            Transformer transformer = TransformerFactory.newInstance().newTransformer();

            transformer.setOutputProperty(OutputKeys.ENCODING, "UTF-8");
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");
            transformer.setOutputProperty("{http:xml.apache.org/xslt}indent-amount", "2");

            DOMSource source = new DOMSource(doc);
            StreamResult result = new StreamResult(file);

            System.out.println("Writing to xml file: " + file.getPath() + " " + file.getName());

            transformer.transform(source, result);
            StreamResult consoleResult = new StreamResult(System.out);
            transformer.transform(source, consoleResult);

        } catch (ParserConfigurationException | DOMException | TransformerException ex) {
            Logger.getLogger(IOManager.class
                    .getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Parses a delimited string into an Int array.
     *
     * <p style="margin-left:5ex;">
     * Courtesy of user MAG of StackExchange.<br>
     * <a href="https://codereview.stackexchange.com/a/122292" target="_blank">
     * https://codereview.stackexchange.com/a/122292</a>
     * </p>
     *
     * @since Java 1.8
     * @param input the delimited input string to be parsed.
     * @param delimiter the delimiter that features in the string.
     * @return an int array containing the parsed values in decimal.
     */
    public static int[] stringToIntArray(String input, String delimiter) {

        return Arrays.stream(input.split(delimiter))
                .mapToInt(Integer::parseInt)
                .toArray();
    }

    /**
     * Parses a delimited string into a Double array.
     *
     * @param input the delimited input string to be parsed.
     * @param delimiter the delimiter that features in the string.
     * @return an double array containing the parsed values in decimal.
     */
    public static double[] stringToDoubleArray(String input, String delimiter) {
        double[] arr = Arrays.stream(input.split(delimiter))
                .mapToDouble(Double::parseDouble)
                .toArray();
        return arr;
    }

}
