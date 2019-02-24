1. build ml module
Run from project root:
`mvn clean install -DskipTests -pl modules/ml -am`

2. Add config. I copypasted from examplse
```
example-default.xml
examples/config/example-ignite.xml
```

3. download the dataset from https://www.kaggle.com/uciml/mushroom-classification#mushrooms.csv

4. Copy-paste DiscreteNaiveBayesTrainerExample.

5. Read dataset. look at `MLSandboxDatasets` as an example.
for simlisity I skip some `?` data.

 
