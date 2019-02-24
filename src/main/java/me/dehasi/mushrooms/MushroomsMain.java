package me.dehasi.mushrooms;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Paths;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.Locale;
import java.util.Scanner;
import java.util.UUID;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.affinity.rendezvous.RendezvousAffinityFunction;
import org.apache.ignite.configuration.CacheConfiguration;
import org.apache.ignite.ml.math.exceptions.knn.FileParsingException;
import org.apache.ignite.ml.math.functions.IgniteBiFunction;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.math.primitives.vector.VectorUtils;
import org.apache.ignite.ml.naivebayes.discrete.DiscreteNaiveBayesModel;
import org.apache.ignite.ml.naivebayes.discrete.DiscreteNaiveBayesTrainer;
import org.apache.ignite.ml.selection.scoring.evaluator.BinaryClassificationEvaluator;

public class MushroomsMain {

    private void run() throws FileNotFoundException {
        System.out.println();
        System.out.println(">>> Discrete naive Bayes classification model over partitioned dataset usage example started.");
        // Start ignite grid.
        try (Ignite ignite = Ignition.start("example-ignite.xml")) {
            System.out.println(">>> Ignite grid started.");

            IgniteCache<Integer, Vector> dataCache = createCacheWith(ignite);

            double[][] thresholds = createThresholds(23);
            System.out.println(">>> Create new Discrete naive Bayes classification trainer object.");
            DiscreteNaiveBayesTrainer trainer = new DiscreteNaiveBayesTrainer()
                .setBucketThresholds(thresholds);

            System.out.println(">>> Perform the training to get the model.");
            IgniteBiFunction<Integer, Vector, Vector> featureExtractor = (k, v) -> v.copyOfRange(1, v.size());
            IgniteBiFunction<Integer, Vector, Double> lbExtractor = (k, v) ->  v.get(0) < 'f'? 1.:0.;

            DiscreteNaiveBayesModel mdl = trainer.fit(
                ignite,
                dataCache,
                featureExtractor,
                lbExtractor
            );

            System.out.println(">>> Discrete Naive Bayes model: " + mdl);

            double accuracy = BinaryClassificationEvaluator.evaluate(
                dataCache,
                mdl,
                featureExtractor,
                lbExtractor
            ).accuracy();

            System.out.println("\n>>> Accuracy " + accuracy);

            System.out.println(">>> Discrete Naive bayes model over partitioned dataset usage example completed.");
        }
    }

    public IgniteCache<Integer, Vector> createCacheWith(Ignite ignite) throws FileNotFoundException {

        IgniteCache<Integer, Vector> cache = getCache(ignite);

        String fileName = "mushrooms.csv";

        ClassLoader classLoader = this.getClass().getClassLoader();
        File file = new File(classLoader.getResource(fileName).getFile());

        if (file == null)
            throw new FileNotFoundException(fileName);

        Scanner scanner = new Scanner(file);
        boolean hasHeader = true;
        int cnt = 0;
        while (scanner.hasNextLine()) {
            String row = scanner.nextLine();
            if (hasHeader && cnt == 0) {
                cnt++;
                continue;
            }

            String[] cells = row.split(",");

            double[] data = new double[cells.length];
            NumberFormat format = NumberFormat.getInstance(Locale.FRANCE);

            for (int i = 0; i < cells.length; i++)
                try {
                    if (cells[i].equals("?") || cells[i].equals(""))
//                        data[i] = Double.NaN;
                        continue;
                    else
                        data[i] = Double.valueOf(cells[i].charAt(0));
                }
                catch (java.lang.NumberFormatException e) {
                    try {
                        data[i] = format.parse(cells[i]).doubleValue();
                    }
                    catch (ParseException e1) {
                        throw new FileParsingException(cells[i], i, Paths.get(fileName));
                    }
                }
            cache.put(cnt++, VectorUtils.of(data));
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
