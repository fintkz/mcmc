// src/pages/api/forecast.js
import { join } from 'path';
import { promises as fs } from 'fs';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const dataPath = join(process.cwd(), 'results/model_results.json');
    const jsonData = await fs.readFile(dataPath, 'utf8');
    const data = JSON.parse(jsonData);
    
    return res.status(200).json(data);
  } catch (error) {
    console.error('Error loading forecast data:', error);
    return res.status(500).json({ error: 'Failed to load forecast data' });
  }
}
