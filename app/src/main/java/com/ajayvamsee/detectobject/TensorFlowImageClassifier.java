package com.ajayvamsee.detectobject;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

public class TensorFlowImageClassifier implements Classifier {

    // Return this many result with at least this confidence
    private static final int MAX_RESULT = 3;
    private static final float THRESHOLD = 0.1f;

    //config values
    private String inputName;
    private String outputName;
    private int inputSize;
    private int imageMean;
    private float imageStd;

    //pre-allocated buffers
    private Vector<String> labels = new Vector<>();
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    private TensorFlowInferenceInterface inferenceInterface;

    private boolean runStats = false;

    private TensorFlowImageClassifier() {
    }

    public static Classifier create(
            AssetManager assetManager,
            String modelFileName,
            String labelFileName,
            int inputSize,
            int imageMean,
            float imageStd,
            String inputName,
            String outputName) throws IOException {

        TensorFlowImageClassifier c = new TensorFlowImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        //Read the labels into memory
        String actualFileName = labelFileName.split("file:///android_asset/")[1];
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(assetManager.open(actualFileName)));
        String line;
        while ((line = br.readLine()) != null) {
            c.labels.add(line);
        }
        br.close();

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFileName);
        // The shape of the output is [N,NUM_CLASSES] , Where N is the batch size.
        int numClasses = (int) c.inferenceInterface.graph().operation(outputName).output(0).shape().size(1);

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputSize = inputSize;
        c.imageMean = imageMean;
        c.imageStd = imageStd;

        // Pre-allocate buffers.
        c.outputNames = new String[]{outputName};
        c.intValues = new int[inputSize * inputSize];
        c.floatValues = new float[inputSize * inputSize * 3];
        c.outputs = new float[numClasses];

        return c;
    }


    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }

        //copy the input data into TensorFLow.
        inferenceInterface.feed(inputName, floatValues, new long[]{1, inputSize, inputSize, 3});

        //Run the interference call
        inferenceInterface.run(outputNames, runStats);

        //copy the output Tensor back into the output array
        inferenceInterface.fetch(outputName, outputs);

        //Find the best classifications
        PriorityQueue<Recognition> pq = new PriorityQueue<>(
                3,
                new Comparator<Recognition>() {
                    @Override
                    public int compare(Recognition o1, Recognition o2) {
                        return Float.compare(o2.getConfidence(), o1.getConfidence());
                    }
                });

        for (int i = 0; i < outputs.length; ++i) {
            if (outputs[i] > THRESHOLD) {
                pq.add(new Recognition("" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
            }
        }

        final ArrayList<Recognition> recognitions=new ArrayList<>();
        int recognitionsSize=Math.min(pq.size(),MAX_RESULT);

        for (int i=0;i<recognitionsSize;++i){
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

    @Override
    public void enableStartLogging(boolean debug) {
        runStats = debug;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
}
