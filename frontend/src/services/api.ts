import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface SensorInput {
  machine_id: string;
  temperature: number;
  vibration: number;
  current: number;
  rpm: number;
  noise: number;
}

export interface PredictResponse {
  machine_id: string;
  health_score: number;
  risk_level: string;
  anomaly: boolean;
  anomaly_score: number;
  failure_type: string;
  failure_probabilities: Record<string, number>;
  rul_hours: number;
}

export const getHealthStatus = () => api.get<{ status: string; service: string }>('/health');
export const getMachines = () => api.get<Array<{ machine_id: string }>>('/machines');
export const predictMachineHealth = (data: SensorInput) => api.post<PredictResponse>('/predict', data);
