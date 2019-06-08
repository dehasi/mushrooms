package me.dehasi.mushrooms;

import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.ml.dataset.feature.extractor.Vectorizer;
import org.apache.ignite.ml.dataset.feature.extractor.impl.DummyVectorizer;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.impl.DenseVector;
import org.apache.ignite.ml.naivebayes.discrete.DiscreteNaiveBayesTrainer;
import org.apache.ignite.ml.pipeline.Pipeline;
import org.apache.ignite.ml.pipeline.PipelineMdl;
import org.apache.ignite.ml.preprocessing.encoding.EncoderTrainer;
import org.apache.ignite.ml.preprocessing.encoding.EncoderType;
import org.apache.ignite.ml.selection.scoring.evaluator.Evaluator;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.UUID;

public class MushroomsMain {

    private void run() throws FileNotFoundException {
        System.out.println();
        System.out.println(">>> Discrete naive Bayes classification model over partitioned dataset usage example started.");
        // Start ignite grid.
        try (Ignite ignite = Ignition.start("example-ignite.xml")) {
            System.out.println(">>> Ignite grid started.");

            IgniteCache<Integer, Vector> dataCache = createCacheWith(ignite);

            int featuresCount = 3;
            double[][] thresholds = createThresholds(featuresCount);
            System.out.println(">>> Create new Discrete naive Bayes classification trainer object.");

            System.out.println(">>> Perform the training to get the model.");


            Vectorizer<Integer, Vector, Integer, String> vectorizer = new MyVectorizer<Integer, String>(1, 2)
                    .zero("").labeled(Vectorizer.LabelCoordinate.FIRST);



            PipelineMdl<Integer, Vector> mdl = new Pipeline<Integer, Vector, Integer, String>()
                    .addVectorizer(vectorizer)
                    .addPreprocessingTrainer(new EncoderTrainer<Integer, Vector>()
                            .withEncoderType(EncoderType.STRING_ENCODER)
                            .withEncodedFeature(0)
                            .withEncodedFeature(1)
                            .withEncodedFeature(2))
                    .addTrainer(new DiscreteNaiveBayesTrainer().setBucketThresholds(thresholds))
                    .fit(ignite, dataCache);


            System.out.println(">>> Discrete Naive Bayes model: " + mdl);

            double accuracy = Evaluator.evaluate(
                    dataCache,
                    mdl,
                    vectorizer
            ).accuracy();

            System.out.println("\n>>> Accuracy " + accuracy);

            System.out.println(">>> Discrete Naive bayes model over partitioned dataset usage example completed.");
        }
    }

    private IgniteCache<Integer, Vector> createCacheWith(Ignite ignite) throws FileNotFoundException {

        IgniteCache<Integer, Vector> cache = getCache(ignite);

        String fileName = "mushrooms_cut.csv";

        ClassLoader classLoader = this.getClass().getClassLoader();
        File file = new File(classLoader.getResource(fileName).getFile());

        if (!file.exists())
            throw new FileNotFoundException(fileName);

        Scanner scanner = new Scanner(file);
        boolean hasHeader = false;
        int cnt = 0;
        while (scanner.hasNextLine()) {
            String row = scanner.nextLine();
            if (hasHeader && cnt == 0) {
                cnt++;
                continue;
            }

            String[] data = row.split(",");


            cache.put(cnt++, new DenseVector(data));
        }
        return cache;

    }

    private IgniteCache<Integer, Vector> getCache(Ignite ignite) {

        CacheConfiguration<Integer, Vector> cacheConfiguration = new CacheConfiguration<>();
        cacheConfiguration.setName("TUTORIAL_" + UUID.randomUUID());
        cacheConfiguration.setAffinity(new RendezvousAffinityFunction(false, 10));

        return ignite.createCache(cacheConfiguration);
    }

    private static double[][] createThresholds(int featuresCount) {
        String alphabet = "abcdefghijklmnopurqtuvwxyz";

        double[] threshold = new double[alphabet.length() - 1];
        for (int i = 0; i < alphabet.length() - 1; i++) {
            threshold[i] = Double.valueOf(alphabet.charAt(i)) + .5;
        }

        double[][] thresholds = new double[featuresCount][];
        for (int i = 0; i < thresholds.length; i++) {
            thresholds[i] = threshold;
        }
        return thresholds;
    }

    public static void main(String[] args) throws FileNotFoundException {
        new MushroomsMain().run();
    }
}
