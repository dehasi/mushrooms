package me.dehasi.mushrooms;

import org.apache.ignite.ml.dataset.DatasetBuilder;
import org.apache.ignite.ml.dataset.feature.extractor.Vectorizer;
import org.apache.ignite.ml.dataset.feature.extractor.impl.DummyVectorizer;
import org.apache.ignite.ml.dataset.impl.local.LocalDatasetBuilder;
import org.apache.ignite.ml.environment.LearningEnvironmentBuilder;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.impl.DenseVector;
import org.apache.ignite.ml.preprocessing.encoding.EncoderPreprocessor;
import org.apache.ignite.ml.preprocessing.encoding.EncoderTrainer;
import org.apache.ignite.ml.preprocessing.encoding.EncoderType;
import org.junit.Test;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;

public class MyVectorizerTest {


    @Test
    public void testFitOnStringCategorialFeatures() {
        Map<Integer, Vector> data = new HashMap<>();
        data.put(1, new DenseVector(new Serializable[]{"Monday", "September"}));
        data.put(2, new DenseVector(new Serializable[]{"Monday", "August"}));
        data.put(3, new DenseVector(new Serializable[]{"Monday", "August"}));
        data.put(4, new DenseVector(new Serializable[]{"Friday", "June"}));
        data.put(5, new DenseVector(new Serializable[]{"Friday", "June"}));
        data.put(6, new DenseVector(new Serializable[]{"Sunday", "August"}));

        final Vectorizer<Integer, Vector, Integer, String> vectorizer = new MyVectorizer<Integer, String>(0 , 1).labeled(0);

        DatasetBuilder<Integer, Vector> datasetBuilder = new LocalDatasetBuilder<>(data, 1);

        EncoderTrainer<Integer, Vector> strEncoderTrainer = new EncoderTrainer<Integer, Vector>()
                .withEncoderType(EncoderType.STRING_ENCODER)
                .withEncodedFeature(0)
                .withEncodedFeature(1);

        EncoderPreprocessor<Integer, Vector> preprocessor = strEncoderTrainer.fit(
                LearningEnvironmentBuilder.defaultBuilder().withRNGSeed(43),
                datasetBuilder,
                vectorizer
        );

        assertArrayEquals(new double[]{0.0, 2.0}, preprocessor.apply(7, new DenseVector(new Serializable[]{"Monday", "September"})).features().asArray(), 1e-8);
    }
}
