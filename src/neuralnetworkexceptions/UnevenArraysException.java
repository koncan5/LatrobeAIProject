/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetworkexceptions;

/**
 *
 * @author konca
 */
public class UnevenArraysException extends Exception {

    /**
     * Creates a new instance of <code>UnevenLinkArraysException</code> without
     * detail message.
     */
    public UnevenArraysException() {
        super("Uneven Arrays Exception: Arrays are of different length");
    }

    /**
     * Constructs an instance of <code>UnevenLinkArraysException</code> with the
     * specified detail message.
     *
     * @param msg the detail message.
     */
    public UnevenArraysException(String msg) {
        super(msg);
    }
}
