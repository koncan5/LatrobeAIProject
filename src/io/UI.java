package io;

/*
 Author: Curtis Alcock 18403879
 Project: Artificial Intelligence Algorithm: Back-Propagation with Accelerated Learning
 Title: UI
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import neuralnetwork.Network;
import neuralnetworkexceptions.UnevenArraysException;

/**
 * Front-End Class: Interacts with the user. Provides user IO for the system.
 * Serves as an interface between the user and the controller.
 *
 * @author Curtis Alcock 18403879
 */
public class UI {

    /**
     * Reads user input from the console.
     */
    public static Scanner c = new Scanner(System.in);

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        Network network;
        double[][] trainingSets;
        double[][] desiredOutcomes;
        String work;

        System.out.println("Welcome to Curtis Alcock's Back Propagation Neural Network.\n");

        System.out.print("Please enter the relative location of the network file: ");
        File file = new File(c.nextLine());
        network = IOManager.readNetwork(file);

        System.out.print("What would you like the network to do (run, train, exit)?: ");
        work = c.nextLine();

        while (!work.equals("exit")) {
            switch (work) {
                case "run": {
                    System.out.print("Please enter the input values you would like to run the network against: i.e. '1,0,1' ");
                    double[] set = IOManager.stringToDoubleArray(c.nextLine(), ",");
                    try {
                        System.out.println(network.run(set));
                    } catch (UnevenArraysException ex) {
                        Logger.getLogger(UI.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    break;
                }
                case "train": {
                    System.out.print("Please enter the relative location of the training sets file: ");
                    file = new File(c.nextLine());
                    trainingSets = IOManager.readTrainingSets(file);
                    desiredOutcomes = IOManager.readDesiredOutcomes(file);

                    System.out.print("Would you like to use accelerated training? (y/n) ");
                    work = c.nextLine();
                    switch (work) {
                        case "y": {
                            try {
                                network.train(trainingSets, desiredOutcomes, 0.0001, false);
                            } catch (UnevenArraysException ex) {
                                Logger.getLogger(UI.class.getName()).log(Level.SEVERE, null, ex);
                            }
                            break;
                        }
                        case "n": {
                            try {
                                network.train(trainingSets, desiredOutcomes, 0.0001, true);
                            } catch (UnevenArraysException ex) {
                                Logger.getLogger(UI.class.getName()).log(Level.SEVERE, null, ex);
                            }
                            break;
                        }
                        default: {
                            System.out.println("Please enter either y/n. ");
                        }
                    }
                    break;
                }

                case "exit": {
                    break;
                }
                default: {
                    System.out.println("Please enter one of the given options. (run, train, exit) ");
                }
            }
            System.out.print("What would you like the network to do (run, train, exit)?: ");
            work = c.nextLine();
        }
    }
}
