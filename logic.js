(function(g) {
  'use strict';

  const app = g.ADKApp || {};

  function normaliseDigits(raw) {
    return String(raw ?? '').replace(/[０-９．]/g, (character) => (
      String.fromCharCode(character.charCodeAt(0) - 0xfee0)
    ));
  }

  app.parseValue = function parseValue(raw) {
    const normalised = normaliseDigits(raw).trim();
    if (normalised === '') return NaN;
    return Number(normalised);
  };

  app.extractNumbers = function extractNumbers(text) {
    const normalised = normaliseDigits(text).replace(/，/g, ',');
    return normalised.match(/\d+(?:[.,]\d+)?/g)?.map((token) => Number(token.replace(',', '.'))) || [];
  };

  app.assignReadings = function assignReadings(numbers, features) {
    if (numbers.length !== 15) return null;
    return Object.fromEntries(features.map((feature, index) => [feature, String(numbers[index].toFixed(1))]));
  };

  app.isFeatureValid = function isFeatureValid(feature, rawValue, featureMeta) {
    const value = app.parseValue(rawValue);
    if (!Number.isFinite(value)) return false;
    if (value < featureMeta.hardMin || value > featureMeta.hardMax) return false;
    if (featureMeta.isRotation && !featureMeta.allowed.some((allowed) => value === Number(allowed))) {
      return false;
    }
    return true;
  };

  app.allValid = function allValid(values, features, featureMeta) {
    return features.every((feature) => app.isFeatureValid(feature, values[feature], featureMeta[feature]));
  };

  app.firstInvalidGroup = function firstInvalidGroup(values, groups, featureMeta) {
    const group = groups.find((item) => !app.allValid(values, item.features, featureMeta));
    return group ? group.number : null;
  };

  app.judge = function judge(values, features, threshold) {
    const numericValues = features.map((feature) => app.parseValue(values[feature]));
    const probability = g.ADKModel.faultProbability(numericValues);
    return probability >= threshold ? 'fault' : 'safe';
  };

  g.ADKApp = app;
})(typeof window !== 'undefined' ? window : globalThis);
