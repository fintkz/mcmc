import { useState, useEffect } from 'react';
import { Card } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const features = [
  { id: 'promotions_active', label: 'Promotions' },
  { id: 'weather_event', label: 'Weather' },
  { id: 'sports_event', label: 'Sports' },
  { id: 'school_term', label: 'School' },
  { id: 'holiday', label: 'Holiday' }
];

const FeatureSelector = ({ features, selectedFeatures, onChange }) => {
  return (
    <div className="flex flex-wrap gap-2 mb-4">
      {features.map(feature => (
        <button
          key={feature.id}
          className={`px-3 py-1 rounded-full text-sm ${
            selectedFeatures.includes(feature.id)
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 text-gray-700'
          }`}
          onClick={() => {
            onChange(prev => 
              prev.includes(feature.id)
                ? prev.filter(f => f !== feature.id)
                : [...prev, feature.id]
            );
          }}
        >
          {feature.label}
        </button>
      ))}
    </div>
  );
};

const ModelPlot = ({ data, modelName, selectedFeatures }) => {
  if (!data?.predictions) return null;

  // Get the correct predictions based on selected features
  const featureKey = selectedFeatures.length > 0 
    ? selectedFeatures
        .sort((a, b) => a.localeCompare(b))
        .join('_')
    : 'baseline';

  // First try exact match
  let modelData = data.predictions[featureKey]?.[modelName];
  let usedKey = featureKey;

  // If exact match not found, try to find the closest combination
  if (!modelData) {
    const availableKeys = Object.keys(data.predictions);
    
    // Filter keys that contain all selected features
    const matchingKeys = availableKeys.filter(key => {
      return selectedFeatures.every(feature => key.includes(feature));
    });

    // Sort by length to get the closest match (fewest additional features)
    matchingKeys.sort((a, b) => a.split('_').length - b.split('_').length);
    
    // Use the first matching combination if available
    if (matchingKeys.length > 0) {
      usedKey = matchingKeys[0];
      modelData = data.predictions[usedKey][modelName];
    }
  }

  const actual = data.actual;

  if (!modelData || !actual) {
    return <Card className="w-full p-4">
      <h2 className="text-xl font-bold mb-4">
        No data available for the selected combination in {modelName.toUpperCase()}
      </h2>
      <p className="text-sm text-gray-600">
        Try a different combination of features
      </p>
    </Card>;
  }

  const chartData = actual.map((val, idx) => ({
    name: idx,
    actual: val,
    predicted: modelData.yhat[idx],
    upper: modelName === 'bayesian' ? modelData.yhat[idx] + 2 * modelData.uncertainty[idx] : null,
    lower: modelName === 'bayesian' ? modelData.yhat[idx] - 2 * modelData.uncertainty[idx] : null
  }));

  return (
    <Card className="w-full p-4">
      <h2 className="text-xl font-bold mb-4">
        {modelName.toUpperCase()} Model 
        {modelData.metrics && ` (MAPE: ${modelData.metrics.mape.toFixed(2)}%)`}
      </h2>
      {featureKey !== usedKey && (
        <p className="text-sm text-amber-600 mb-2">
          {/* Note: Showing predictions for combination: {usedKey.split('_').join(' + ')} */}
        </p>
      )}
      <div className="h-[400px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="name" 
              label={{ value: 'Time', position: 'bottom' }}
            />
            <YAxis 
              label={{ value: 'Demand', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="actual" 
              stroke="#000000" 
              name="Actual" 
              dot={false}
            />
            <Line 
              type="monotone" 
              dataKey="predicted" 
              stroke="#0088FE" 
              name="Predicted" 
              dot={false}
            />
            {modelName === 'bayesian' && (
              <>
                <Line 
                  type="monotone" 
                  dataKey="upper" 
                  stroke="transparent" 
                  fill="#0088FE" 
                  fillOpacity={0.1}
                />
                <Line 
                  type="monotone" 
                  dataKey="lower" 
                  stroke="transparent" 
                  fill="#0088FE" 
                  fillOpacity={0.1}
                />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
};

export default function Home() {
  const [data, setData] = useState(null);
  const [selectedFeatures1, setSelectedFeatures1] = useState([]);
  const [selectedFeatures2, setSelectedFeatures2] = useState([]);
  const [selectedFeatures3, setSelectedFeatures3] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/api/forecast');
        if (!response.ok) throw new Error('Failed to fetch data');
        const result = await response.json();
        setData(result);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <div className="p-4">Loading...</div>;
  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!data) return <div className="p-4">No data available</div>;

  return (
    <main className="container mx-auto p-4 max-w-[1400px]">
      <h1 className="text-3xl font-bold mb-6">Demand Forecasting Model Comparison</h1>
      
      <div className="flex flex-col gap-8">
        {/* Prophet Model */}
        <section>
          <h2 className="text-xl font-bold mb-2">Prophet Model</h2>
          <FeatureSelector 
            features={features} 
            selectedFeatures={selectedFeatures1} 
            onChange={setSelectedFeatures1}
          />
          <ModelPlot 
            data={data} 
            modelName="prophet" 
            selectedFeatures={selectedFeatures1} 
          />
        </section>

        {/* TFT Model */}
        <section>
          <h2 className="text-xl font-bold mb-2">Temporal Fusion Transformer (TFT)</h2>
          <FeatureSelector 
            features={features} 
            selectedFeatures={selectedFeatures2} 
            onChange={setSelectedFeatures2}
          />
          <ModelPlot 
            data={data} 
            modelName="tft" 
            selectedFeatures={selectedFeatures2} 
          />
        </section>

        {/* Bayesian Model */}
        <section>
          <h2 className="text-xl font-bold mb-2">Bayesian Ensemble</h2>
          <FeatureSelector 
            features={features} 
            selectedFeatures={selectedFeatures3} 
            onChange={setSelectedFeatures3}
          />
          <ModelPlot 
            data={data} 
            modelName="bayesian" 
            selectedFeatures={selectedFeatures3} 
          />
        </section>
      </div>

      <Card className="p-4 mt-8">
        <h2 className="text-xl font-bold mb-4">Model Characteristics</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h3 className="font-bold mb-2">Prophet</h3>
            <ul className="list-disc pl-4 space-y-1">
              <li>Handles holidays and seasonality explicitly</li>
              <li>Decomposes trend, seasonality, and holiday effects</li>
              <li>Best for data with strong seasonal patterns</li>
            </ul>
          </div>
          <div>
            <h3 className="font-bold mb-2">Temporal Fusion Transformer (TFT)</h3>
            <ul className="list-disc pl-4 space-y-1">
              <li>Captures complex feature interactions</li>
              <li>Uses attention mechanism for interpretability</li>
              <li>Excels at long-term dependencies</li>
            </ul>
          </div>
          <div>
            <h3 className="font-bold mb-2">Bayesian Ensemble</h3>
            <ul className="list-disc pl-4 space-y-1">
              <li>Provides uncertainty estimates</li>
              <li>Robust to outliers and noise</li>
              <li>Combines multiple models for better predictions</li>
            </ul>
          </div>
        </div>
      </Card>
    </main>
  );
}