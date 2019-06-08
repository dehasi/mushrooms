package me.dehasi.mushrooms;

import org.apache.ignite.ml.dataset.feature.extractor.Vectorizer;
import org.apache.ignite.ml.math.primitives.vector.Vector;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MyVectorizer<KEY, LABEL> extends Vectorizer<KEY, Vector, Integer, LABEL> {

    private LABEL zero;

    public MyVectorizer(Integer... coords) {
        super(coords);
    }

    @Override
    protected Serializable feature(Integer coord, KEY key, Vector value) {
        return value.getRawX(coord);
    }

    @Override
    protected LABEL label(Integer coord, KEY key, Vector value) {
        return (LABEL)feature(coord, key, value);
    }

    @Override
    protected LABEL zero() {
        return zero;
    }

    @Override
    protected List<Integer> allCoords(KEY key, Vector value) {
        return IntStream.range(0, sizeOf(key, value)).boxed().collect(Collectors.toList());
    }

    public MyVectorizer<KEY, LABEL> zero(LABEL zero) {
        this.zero = zero;
        return this;
    }

    protected int sizeOf(KEY key, Vector value) {
        return value.size();
    }

}
