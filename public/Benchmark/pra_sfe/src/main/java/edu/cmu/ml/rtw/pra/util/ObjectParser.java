package edu.cmu.ml.rtw.pra.util;

/**
 * Parses objects of type T from strings.
 */
public interface ObjectParser<T> {
  public T fromString(String string);
}
