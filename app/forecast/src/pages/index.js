import { useState, useEffect } from 'react';
import { Card } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Feature mapping
const features = [
  { id: '0', name: 'Promotions', description: 'Sales and promotional events' },
  { id: '1', name: 'Weather', description: 'Weather events and conditions' },
  { id: '2', name: 'Sports', description: 'Sports events' },
  { id: '3', name: 'School', description: 'School terms and holidays' },
  { id: '4', name: 'Holidays', description: 'Public and seasonal holidays' }
];

// Valid feature combinations
const validCombinations = [
  '0', '0_1', '0_1_2', '0_1_2_3', '0_1_2_3_4', '0_1_2_4', '0_1_3', '0_1_3_4',
  '0_1_4', '0_2', '0_2_3', '0_2_3_4', '0_2_4', '0_3', '0_3_4', '0_4',
  '1', '1_2', '1_2_3', '1_2_3_4', '1_2_4', '1_3', '1_3_4', '1_4',
  '2', '2_3', '2_3_4', '2_4', '3', '3_4', '4', 'baseline'
];

const FeatureSelector = ({ features, selectedFeatures, onChange }) => {
  return (
    <div className="flex flex-wrap gap-2 mb-4">
      {features.map(feature => (
        <button
          key={feature.id}
          className={`px-3 py-1 rounded-full text-sm ${selectedFeatures.includes(feature.id)
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
          {feature.name}
        </button>
      ))}
    </div>
  );
};

const ModelPlot = ({ data, modelName, selectedFeatures }) => {
  if (!data?.predictions) return null;

  // Get the correct predictions based on selected features
  const selectedKey = selectedFeatures.length > 0
    ? selectedFeatures
      .map(Number)
      .sort((a, b) => a - b)
      .join('_')
    : 'baseline';

  // Find the exact match or return baseline
  let modelData = data.predictions[selectedKey]?.[modelName];
  let usedKey = selectedKey;

  // If exact match not found, use baseline
  if (!modelData) {
    modelData = data.predictions['baseline']?.[modelName];
    usedKey = 'baseline';
  }

  // Return early if no valid data is found
  if (!modelData?.predictions || !Array.isArray(modelData.predictions) || !Array.isArray(data.actual)) {
    return (
      <div className="text-center p-4">
        No predictions available for the selected features
      </div>
    );
  }

  // Combine actual and predicted values
  const chartData = modelData.predictions.map((pred, idx) => {
    const dataPoint = {
      timestamp: idx,
      actual: data.actual[idx],
      predicted: pred
    };

    // Add uncertainty bounds for bayesian model
    if (modelName === 'bayesian' && Array.isArray(modelData.uncertainty)) {
      dataPoint.lower_bound = pred - modelData.uncertainty[idx];
      dataPoint.upper_bound = pred + modelData.uncertainty[idx];
    }

    return dataPoint;
  });

  // Add metrics if available
  const metrics = modelData.metrics ? {
    mape: modelData.metrics.mape.toFixed(2),
    rmse: modelData.metrics.rmse.toFixed(2)
  } : null;

  return (
    <Card className="w-full p-4">
      <h2 className="text-xl font-bold mb-2">
        {modelName.toUpperCase()} Model
        {metrics && ` (MAPE: ${metrics.mape}%, RMSE: ${metrics.rmse})`}
      </h2>
      {selectedKey !== usedKey && (
        <p className="text-sm text-amber-600 mb-2">
          Using closest available feature combination: {usedKey}
        </p>
      )}
      <div className="w-full h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="timestamp"
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
                  dataKey="lower_bound"
                  stroke="transparent"
                  fill="#0088FE"
                  fillOpacity={0.1}
                />
                <Line
                  type="monotone"
                  dataKey="upper_bound"
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