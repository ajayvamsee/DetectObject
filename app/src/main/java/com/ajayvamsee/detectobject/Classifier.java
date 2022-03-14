package com.ajayvamsee.detectobject;

import android.graphics.Bitmap;
import android.graphics.RectF;

import java.util.List;

public interface Classifier {

    // An Immutable interface result returned by a classifier describing what was recognized

    public class Recognition {

        // A unique identifier for what has been recognized. Specific to the class not the instance of the object
        private final String id;

        //display name for the Recognition
        private final String title;

        // A Sortable score for how good the recognition is relative to others. Higher should be better
        private final Float confidence;

        //Optimal Location within the source image for the location of the recognized object
        private RectF location;

        public Recognition(String id, String title, Float confidence,final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location=location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return location;
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }

    List<Recognition> recognizeBitMap(Bitmap bitmap);

    void enableStartLogging(final boolean debug);

    String getStatString();

    void close();
}
