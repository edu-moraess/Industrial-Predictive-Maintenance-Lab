import React, { useEffect, useState, useCallback } from 'react';
import { 
  Activity, AlertTriangle, ShieldCheck, Clock, RefreshCw, Cpu, 
  Server, Flame, Zap, Gauge, Volume2, Thermometer, Radio
} from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  BarChart, Bar, CartesianGrid, Cell 
} from 'recharts';
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
  const [lastSensorValues, setLastSensorValues] = useState<SensorInput | null>(null);

  const generateSensorData = useCallback((): SensorInput => {
    if (isSimulatingFailure) {
      return {
        machine_id: machineId,
        temperature: 82.3 + Math.random() * 8,
        vibration: 7.8 + Math.random() * 2.5,
        current: 28.4 + Math.random() * 4,
        rpm: 1610.0 + Math.random() * 40,
        noise: 89.2 + Math.random() * 6,
      };
    }

    return {
      machine_id: machineId,
      temperature: 42.5 + Math.random() * 4,
      vibration: 1.8 + Math.random() * 0.6,
      current: 14.8 + Math.random() * 1.2,
      rpm: 1792.0 + Math.random() * 15,
      noise: 54.1 + Math.random() * 3,
    };
  }, [machineId, isSimulatingFailure]);

  const fetchPrediction = useCallback(async () => {
    try {
      const payload = generateSensorData();
      setLastSensorValues(payload);
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
      setError('Conexão perdida com o motor de inferência FastAPI.');
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
      <div className="flex flex-col items-center justify-center h-screen bg-[#070A0F] text-cyan-400 font-mono space-y-4">
        <RefreshCw className="animate-spin w-8 h-8 text-cyan-400" />
        <div className="tracking-widest text-xs uppercase">Carregando Telemetria do Ativo...</div>
      </div>
    );
  }

  const probData = data
    ? Object.entries(data.failure_probabilities).map(([name, val]) => ({
        name: name.replace('_', ' '),
        prob: val,
      }))
    : [];

  const getRiskColor = (level?: string) => {
    switch (level?.toUpperCase()) {
      case 'NORMAL': case 'SAUDÁVEL': case 'LOW': return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
      case 'MEDIUM': case 'ALERTA': return 'text-amber-400 border-amber-500/30 bg-amber-500/10';
      default: return 'text-rose-400 border-rose-500/30 bg-rose-500/10';
    }
  };

  return (
    <div className="min-h-screen bg-[#070A0F] text-slate-200 font-sans p-4 md:p-6 space-y-6">
      {/* SCADA Header */}
      <header className="flex flex-col lg:flex-row lg:items-center justify-between border-b border-slate-800/80 pb-5 gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-slate-900 border border-slate-700/60 rounded-xl shadow-inner">
            <Radio className="text-cyan-400 w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-wider text-white uppercase font-mono">SCADA Control Room</h1>
              <span className="text-[10px] font-mono bg-cyan-950 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded">
                UNIT-{machineId}
              </span>
            </div>
            <p className="text-xs text-slate-400">Monitoramento Preditivo Contínuo & Matriz de Inferência ML</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsSimulatingFailure(!isSimulatingFailure)}
            className={`px-4 py-2 rounded-lg text-xs font-mono font-bold flex items-center gap-2 transition-all border ${
              isSimulatingFailure
                ? 'bg-rose-950/80 text-rose-400 border-rose-600 animate-pulse shadow-lg shadow-rose-950/50'
                : 'bg-slate-900 text-slate-300 border-slate-700 hover:bg-slate-800 hover:border-slate-600'
            }`}
          >
            <Flame className="w-4 h-4 text-rose-400" />
            {isSimulatingFailure ? 'INJEÇÃO DE FALHA ATIVA' : 'INJETAR FALHA'}
          </button>

          <div className="flex items-center gap-2 bg-slate-900/90 border border-slate-800 px-3 py-2 rounded-lg">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping"></span>
            <span className="text-xs font-mono text-emerald-400">API ONLINE</span>
          </div>
        </div>
      </header>

      {error && (
        <div className="bg-rose-950/40 border border-rose-800/60 text-rose-300 px-4 py-3 rounded-xl text-xs font-mono flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* KPI Section */}
      {data && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl relative overflow-hidden backdrop-blur-md">
            <div className="flex justify-between items-start text-slate-400">
              <span className="text-[11px] font-mono font-semibold tracking-wider text-slate-400 uppercase">Integridade (Health Score)</span>
              <Activity className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-3xl font-extrabold font-mono text-white">{data.health_score.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
              <div 
                className={`h-full transition-all duration-500 ${data.health_score > 70 ? 'bg-cyan-400' : 'bg-rose-500'}`} 
                style={{ width: `${data.health_score}%` }} 
              />
            </div>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl backdrop-blur-md">
            <div className="flex justify-between items-start text-slate-400">
              <span className="text-[11px] font-mono font-semibold tracking-wider text-slate-400 uppercase">Projeção RUL</span>
              <Clock className="w-4 h-4 text-purple-400" />
            </div>
            <div className="mt-2">
              <span className="text-3xl font-extrabold font-mono text-purple-300">{data.rul_hours.toFixed(0)} <span className="text-xs text-purple-400/70">HRS</span></span>
            </div>
            <p className="text-[10px] text-slate-500 mt-2">Vida útil remanescente estimada</p>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl backdrop-blur-md">
            <div className="flex justify-between items-start text-slate-400">
              <span className="text-[11px] font-mono font-semibold tracking-wider text-slate-400 uppercase">Classificação de Risco</span>
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="mt-2">
              <span className={`inline-block text-xs font-mono font-bold px-2.5 py-1 rounded-md border ${getRiskColor(data.risk_level)}`}>
                {data.risk_level}
              </span>
            </div>
            <p className="text-[10px] text-slate-500 mt-2">Status do modelo preditivo</p>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl backdrop-blur-md">
            <div className="flex justify-between items-start text-slate-400">
              <span className="text-[11px] font-mono font-semibold tracking-wider text-slate-400 uppercase">Detecção Isolation Forest</span>
              <AlertTriangle className={`w-4 h-4 ${data.anomaly ? 'text-rose-400' : 'text-slate-600'}`} />
            </div>
            <div className="mt-2">
              <span className={`text-xl font-bold font-mono ${data.anomaly ? 'text-rose-400' : 'text-emerald-400'}`}>
                {data.anomaly ? 'ANOMALIA CRÍTICA' : 'OPERANDO NORMAL'}
              </span>
            </div>
            <p className="text-[10px] text-slate-500 mt-2">Score: {data.anomaly_score.toFixed(3)}</p>
          </div>
        </div>
      )}

      {/* Live Sensors Readouts Grid */}
      {lastSensorValues && (
        <div className="space-y-2">
          <h2 className="text-xs font-mono uppercase tracking-wider text-slate-400">Telemetria de Campo Instantânea</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <div className="bg-slate-900/40 border border-slate-800/80 p-3 rounded-lg flex items-center gap-3">
              <Thermometer className="w-5 h-5 text-cyan-400 shrink-0" />
              <div>
                <div className="text-[10px] text-slate-400 font-mono">TEMP.</div>
                <div className="text-sm font-bold font-mono text-slate-100">{lastSensorValues.temperature.toFixed(1)} °C</div>
              </div>
            </div>

            <div className="bg-slate-900/40 border border-slate-800/80 p-3 rounded-lg flex items-center gap-3">
              <Activity className="w-5 h-5 text-amber-400 shrink-0" />
              <div>
                <div className="text-[10px] text-slate-400 font-mono">VIBRAÇÃO</div>
                <div className="text-sm font-bold font-mono text-slate-100">{lastSensorValues.vibration.toFixed(2)} mm/s</div>
              </div>
            </div>

            <div className="bg-slate-900/40 border border-slate-800/80 p-3 rounded-lg flex items-center gap-3">
              <Zap className="w-5 h-5 text-blue-400 shrink-0" />
              <div>
                <div className="text-[10px] text-slate-400 font-mono">CORRENTE</div>
                <div className="text-sm font-bold font-mono text-slate-100">{lastSensorValues.current.toFixed(1)} A</div>
              </div>
            </div>

            <div className="bg-slate-900/40 border border-slate-800/80 p-3 rounded-lg flex items-center gap-3">
              <Gauge className="w-5 h-5 text-emerald-400 shrink-0" />
              <div>
                <div className="text-[10px] text-slate-400 font-mono">ROTAÇÃO</div>
                <div className="text-sm font-bold font-mono text-slate-100">{lastSensorValues.rpm.toFixed(0)} RPM</div>
              </div>
            </div>

            <div className="bg-slate-900/40 border border-slate-800/80 p-3 rounded-lg flex items-center gap-3 col-span-2 sm:col-span-1">
              <Volume2 className="w-5 h-5 text-purple-400 shrink-0" />
              <div>
                <div className="text-[10px] text-slate-400 font-mono">RUÍDO</div>
                <div className="text-sm font-bold font-mono text-slate-100">{lastSensorValues.noise.toFixed(1)} dB</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Server className="w-4 h-4 text-cyan-400" /> Tendência Dinâmica de Sensores
            </h2>
          </div>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%" debounce={50}>
              <AreaChart key={`area-${history.length}`} data={history}>
                <defs>
                  <linearGradient id="tempColor" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06B6D4" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#06B6D4" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="vibColor" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#F59E0B" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="time" stroke="#64748B" fontSize={10} tickLine={false} />
                <YAxis stroke="#64748B" fontSize={10} tickLine={false} />
                <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', color: '#F8FAFC', fontSize: '12px' }} />
                <Area type="monotone" dataKey="temperature" name="Temp (°C)" stroke="#06B6D4" fillOpacity={1} fill="url(#tempColor)" isAnimationActive={false} />
                <Area type="monotone" dataKey="vibration" name="Vibração (mm/s)" stroke="#F59E0B" fillOpacity={1} fill="url(#vibColor)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" /> Probabilidade de Modos de Falha (Random Forest)
            </h2>
          </div>
          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%" debounce={50}>
              <BarChart key={`bar-${probData.length}`} data={probData} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" horizontal={false} />
                <XAxis type="number" stroke="#64748B" domain={[0, 100]} fontSize={10} tickLine={false} />
                <YAxis dataKey="name" type="category" stroke="#94A3B8" width={110} fontSize={9} tickLine={false} />
                <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', color: '#F8FAFC', fontSize: '12px' }} />
                <Bar dataKey="prob" name="Probabilidade (%)" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                  {probData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.prob > 40 ? '#F43F5E' : '#3B82F6'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
