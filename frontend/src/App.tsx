import React, { useEffect, useState, useCallback } from 'react';
import { Activity, AlertTriangle, ShieldCheck, Clock, RefreshCw, Cpu, Server, Flame, ActivitySquare } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { predictMachineHealth, PredictResponse, SensorInput } from './services/api';

interface TelemetryPoint {
  time: string;
  temperature: number;
  vibration: number;
  health: number;
}

export default function App() {
  const [data, setData] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<TelemetryPoint[]>([]);
  const [machineId] = useState<string>('MACHINE_001');
  const [isSimulatingFailure, setIsSimulatingFailure] = useState<boolean>(false);

  const generateSensorData = useCallback((): SensorInput => {
    if (isSimulatingFailure) {
      return {
        machine_id: machineId,
        temperature: 78.5 + Math.random() * 12,
        vibration: 7.2 + Math.random() * 3.5,
        current: 26.0 + Math.random() * 6,
        rpm: 1650.0 + Math.random() * 50,
        noise: 88.0 + Math.random() * 8,
      };
    }

    return {
      machine_id: machineId,
      temperature: 42.0 + Math.random() * 8,
      vibration: 1.8 + Math.random() * 1.2,
      current: 14.5 + Math.random() * 2,
      rpm: 1790.0 + Math.random() * 20,
      noise: 64.0 + Math.random() * 5,
    };
  }, [machineId, isSimulatingFailure]);

  const fetchPrediction = useCallback(async () => {
    try {
      const payload = generateSensorData();
      const response = await predictMachineHealth(payload);
      
      setData(response.data);
      setError(null);

      const timestamp = new Date().toLocaleTimeString('pt-BR', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
      });

      setHistory((prev) => [
        ...prev.slice(-19),
        {
          time: timestamp,
          temperature: parseFloat(payload.temperature.toFixed(1)),
          vibration: parseFloat(payload.vibration.toFixed(2)),
          health: response.data.health_score,
        },
      ]);
    } catch (err: any) {
      setError('Falha de comunicação com a API REST FastAPI.');
    } finally {
      setLoading(false);
    }
  }, [generateSensorData]);

  useEffect(() => {
    fetchPrediction();
    const interval = setInterval(fetchPrediction, 2500);
    return () => clearInterval(interval);
  }, [fetchPrediction]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-industrial-bg text-industrial-accent font-mono">
        <RefreshCw className="animate-spin mr-3 w-6 h-6" /> INICIALIZANDO INTERFACE SCADA...
      </div>
    );
  }

  const probData = data
    ? Object.entries(data.failure_probabilities).map(([name, val]) => ({
        name,
        prob: val,
      }))
    : [];

  return (
    <div className="min-h-screen p-6 space-y-6 bg-industrial-bg text-slate-100 font-sans">
      {/* Top Header */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-industrial-border pb-4 gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-industrial-card rounded-lg border border-industrial-border">
            <Cpu className="text-industrial-accent w-7 h-7" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-wider text-white">INDUSTRIAL SCADA MONITOR</h1>
            <p className="text-xs text-slate-400 font-mono">Backend Engine: FastAPI + PyTorch/Scikit-Learn</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => setIsSimulatingFailure(!isSimulatingFailure)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all border ${
              isSimulatingFailure
                ? 'bg-rose-500/20 text-rose-400 border-rose-500/50 animate-pulse'
                : 'bg-industrial-card text-slate-300 border-industrial-border hover:border-slate-500'
            }`}
          >
            <Flame className="w-4 h-4" />
            {isSimulatingFailure ? 'Injetando Falha Crítica' : 'Simular Anomalia'}
          </button>

          <div className="flex items-center gap-2 bg-industrial-card px-3 py-1.5 rounded-lg border border-industrial-border">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-ping"></span>
            <span className="text-xs text-slate-300 font-mono">LIVE API FEED</span>
          </div>
        </div>
      </header>

      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 text-rose-400 p-3 rounded-lg text-sm flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {/* KPI Display */}
      {data && (
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="bg-industrial-card border border-industrial-border p-4 rounded-xl shadow-lg">
            <div className="flex justify-between items-center text-slate-400 mb-2">
              <span className="text-xs font-semibold tracking-wide">HEALTH SCORE</span>
              <Activity className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-3xl font-extrabold text-white">{data.health_score}%</div>
          </div>

          <div className="bg-industrial-card border border-industrial-border p-4 rounded-xl shadow-lg">
            <div className="flex justify-between items-center text-slate-400 mb-2">
              <span className="text-xs font-semibold tracking-wide">NÍVEL DE RISCO</span>
              <ShieldCheck className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="text-2xl font-bold text-cyan-400">{data.risk_level}</div>
          </div>

          <div className="bg-industrial-card border border-industrial-border p-4 rounded-xl shadow-lg">
            <div className="flex justify-between items-center text-slate-400 mb-2">
              <span className="text-xs font-semibold tracking-wide">ANOMALIA</span>
              <AlertTriangle className={`w-4 h-4 ${data.anomaly ? 'text-rose-500' : 'text-slate-500'}`} />
            </div>
            <div className={`text-xl font-bold ${data.anomaly ? 'text-rose-500' : 'text-emerald-400'}`}>
              {data.anomaly ? 'DETECTADA' : 'NORMAL'}
            </div>
          </div>

          <div className="bg-industrial-card border border-industrial-border p-4 rounded-xl shadow-lg">
            <div className="flex justify-between items-center text-slate-400 mb-2">
              <span className="text-xs font-semibold tracking-wide">DIAGNÓSTICO</span>
              <ActivitySquare className="w-4 h-4 text-amber-400" />
            </div>
            <div className="text-sm font-bold text-amber-400 truncate">{data.failure_type}</div>
          </div>

          <div className="bg-industrial-card border border-industrial-border p-4 rounded-xl shadow-lg">
            <div className="flex justify-between items-center text-slate-400 mb-2">
              <span className="text-xs font-semibold tracking-wide">ESTIMATIVA RUL</span>
              <Clock className="w-4 h-4 text-purple-400" />
            </div>
            <div className="text-2xl font-bold text-purple-400">{data.rul_hours} hrs</div>
          </div>
        </div>
      )}

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-industrial-card border border-industrial-border p-5 rounded-xl shadow-lg">
          <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
            <Server className="w-4 h-4 text-industrial-accent" /> Telemetria em Tempo Real (Vibração & Temp)
          </h2>
          <div className="h-64 w-full">
            <ResponsiveContainer width="99%" height="100%" debounce={50}>
              <AreaChart key={`area-${history.length}`} data={history}>
                <XAxis dataKey="time" stroke="#475569" fontSize={11} />
                <YAxis stroke="#475569" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: '#151C28', borderColor: '#2A3447', color: '#fff' }} />
                <Area type="monotone" dataKey="temperature" name="Temperatura (°C)" stroke="#00E5FF" fill="#00E5FF" fillOpacity={0.15} isAnimationActive={false} />
                <Area type="monotone" dataKey="vibration" name="Vibração (mm/s)" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.15} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-industrial-card border border-industrial-border p-5 rounded-xl shadow-lg">
          <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-industrial-accent" /> Matriz de Probabilidade de Falha (Random Forest)
          </h2>
          <div className="h-64 w-full">
            <ResponsiveContainer width="99%" height="100%" debounce={50}>
              <BarChart key={`bar-${probData.length}`} data={probData} layout="vertical">
                <XAxis type="number" stroke="#475569" domain={[0, 100]} fontSize={11} />
                <YAxis dataKey="name" type="category" stroke="#475569" width={140} fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: '#151C28', borderColor: '#2A3447', color: '#fff' }} />
                <Bar dataKey="prob" name="Probabilidade (%)" fill="#3B82F6" radius={[0, 4, 4, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
