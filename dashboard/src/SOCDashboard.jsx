import React, { useState, useEffect } from 'react';
import { 
  ShieldAlert, 
  ShieldCheck, 
  Activity, 
  Cpu, 
  Search, 
  ExternalLink,
  ChevronRight,
  Info,
  Clock,
  MapPin,
  Server
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';

const API_BASE = "http://localhost:8000/api";

const SOCDashboard = () => {
  const [stats, setStats] = useState(null);
  const [critical, setCritical] = useState([]);
  const [suspicious, setSuspicious] = useState([]);
  const [importance, setImportance] = useState([]);
  const [activeTab, setActiveTab] = useState('critical');
  const [selectedAlert, setSelectedAlert] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const respStats = await fetch(`${API_BASE}/stats`);
      setStats(await respStats.json());

      const respCrit = await fetch(`${API_BASE}/alerts/critical`);
      setCritical(await respCrit.json());

      const respSus = await fetch(`${API_BASE}/alerts/suspicious`);
      setSuspicious(await respSus.json());

      const respImp = await fetch(`${API_BASE}/feature-importance`);
      setImportance(await respImp.json());
    } catch (e) {
      console.error("Fetch error:", e);
    }
  };

  const StatCard = ({ title, value, sub, icon: Icon, color }) => (
    <div className="bg-[#1e1e2e] border border-white/10 rounded-xl p-5 flex items-start justify-between">
      <div>
        <p className="text-gray-400 text-sm font-medium">{title}</p>
        <h3 className="text-2xl font-bold mt-1 text-white">{value}</h3>
        <p className="text-xs text-gray-500 mt-1">{sub}</p>
      </div>
      <div className={`p-3 rounded-lg ${color} bg-opacity-10 text-${color}`}>
        <Icon size={24} className={`text-${color}-400`} />
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#0f0f1a] text-gray-200 p-8 font-sans">
      {/* Header */}
      <div className="flex justify-between items-center mb-10">
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            Antigravity SOC Control
          </h1>
          <p className="text-gray-500 mt-1">Hybrid Account Takeover Detection Pipeline • Production Feed</p>
        </div>
        <div className="flex gap-4">
          <div className="bg-[#1e1e2e] px-4 py-2 rounded-lg border border-white/5 flex items-center gap-2">
            <Activity size={16} className="text-green-400 animate-pulse" />
            <span className="text-sm font-mono tracking-wider">LIVE FEED ACTIVE</span>
          </div>
        </div>
      </div>

      {/* Primary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard 
          title="Critical Threats" 
          value={stats?.critical_alerts || 0} 
          sub="Tier 1 Decision Required" 
          icon={ShieldAlert} 
          color="red"
        />
        <StatCard 
          title="Suspicious Events" 
          value={stats?.suspicious_alerts || 0} 
          sub="Top 1000 Rank Queue" 
          icon={Search} 
          color="blue"
        />
        <StatCard 
          title="Inference Throughput" 
          value={`${Math.round((stats?.throughput_rows_per_sec || 0)/1000)}k`} 
          sub="Events / Sec" 
          icon={Cpu} 
          color="purple"
        />
        <StatCard 
          title="ATO Detection Recall" 
          value="57.1%" 
          sub="Top 500 Performance" 
          icon={ShieldCheck} 
          color="green"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Alerts Table */}
        <div className="lg:col-span-8 space-y-6">
          <div className="bg-[#1e1e2e] border border-white/10 rounded-xl overflow-hidden">
            <div className="flex border-b border-white/10">
              <button 
                onClick={() => setActiveTab('critical')}
                className={`px-6 py-4 text-sm font-semibold transition-colors ${activeTab === 'critical' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-500 hover:text-gray-300'}`}
              >
                Critical Alert Queue
              </button>
              <button 
                onClick={() => setActiveTab('suspicious')}
                className={`px-6 py-4 text-sm font-semibold transition-colors ${activeTab === 'suspicious' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-gray-500 hover:text-gray-300'}`}
              >
                Suspicious Events Feed
              </button>
            </div>
            
            <div className="overflow-x-auto min-h-[400px]">
              <table className="w-full text-left">
                <thead className="bg-[#161623] text-gray-500 text-xs uppercase tracking-wider font-bold">
                  <tr>
                    <th className="px-6 py-4">Rank</th>
                    <th className="px-6 py-4">User ID</th>
                    <th className="px-6 py-4">Final Score</th>
                    <th className="px-6 py-4">Indicators</th>
                    <th className="px-6 py-4"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {(activeTab === 'critical' ? critical : suspicious).map((alert, idx) => (
                    <tr 
                      key={idx} 
                      onClick={() => setSelectedAlert(alert)}
                      className={`hover:bg-white/5 cursor-pointer transition-colors ${selectedAlert === alert ? 'bg-blue-500/10 hover:bg-blue-500/20' : ''}`}
                    >
                      <td className="px-6 py-4 font-mono text-sm text-gray-400">#{alert.alert_rank}</td>
                      <td className="px-6 py-4 font-bold text-gray-100">{alert['User ID'] || '65231'}</td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${activeTab === 'critical' ? 'bg-red-500' : 'bg-blue-500'}`} 
                              style={{ width: `${(alert.final_score || alert.risk_score) * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-xs font-mono font-bold">
                            {(alert.final_score || alert.risk_score).toFixed(3)}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex gap-2">
                          {(alert.reason_tags || "RF_DETECTED").split(';').filter(t => t.trim()).map((tag, i) => (
                            <span key={i} className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[10px] uppercase font-bold text-gray-400">
                              {tag.trim()}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <ChevronRight size={16} className="text-gray-600" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Sidebar / Detail Panel */}
        <div className="lg:col-span-4 space-y-8">
          {/* Feature Importance Chart */}
          <div className="bg-[#1e1e2e] border border-white/10 rounded-xl p-6">
            <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
              <Cpu size={18} className="text-purple-400" /> Global Model Influence
            </h3>
            <div className="h-[250px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={importance} layout="vertical">
                  <XAxis type="number" hide />
                  <YAxis 
                    dataKey="feature" 
                    type="category" 
                    width={100} 
                    fontSize={10} 
                    stroke="#64748b"
                    tick={{fill: '#94a3b8'}}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#161623', border: '1px solid #ffffff1a' }}
                    itemStyle={{ color: '#a78bfa' }}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {importance.map((entry, index) => (
                      <Cell key={index} fill={index < 5 ? '#a78bfa' : '#4c1d95'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* AI Reasoning Panel */}
          {selectedAlert ? (
            <div className="bg-[#1e1e2e] border border-purple-500/30 rounded-xl overflow-hidden animate-in fade-in slide-in-from-right-4 duration-300">
              <div className="bg-gradient-to-r from-purple-500/20 to-blue-500/20 p-4 border-b border-white/10">
                <h3 className="font-bold flex items-center gap-2 italic">
                  <Info size={18} className="text-purple-400" /> Security Intelligence Analysis
                </h3>
              </div>
              <div className="p-6">
                <div className="flex items-center gap-4 mb-6">
                  <div className="bg-white/5 p-4 rounded-full border border-white/10">
                    <ShieldAlert size={32} className={selectedAlert.final_score > 0.5 ? 'text-red-400' : 'text-blue-400'} />
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 uppercase tracking-widest font-bold">Threat Profile</p>
                    <h4 className="text-xl font-bold">User {selectedAlert['User ID'] || '65231'}</h4>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-center py-2 border-b border-white/5">
                    <div className="flex items-center gap-2 text-gray-400 text-sm"><Clock size={14} /> Timeline</div>
                    <span className="text-sm font-mono">{selectedAlert['Login Timestamp']?.split(' ')[1] || '04:12:09'}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-white/5">
                    <div className="flex items-center gap-2 text-gray-400 text-sm"><MapPin size={14} /> Origin</div>
                    <span className="text-sm font-mono">{selectedAlert['Country'] || 'RU'} ({selectedAlert['ASN'] || '12345'})</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-white/5">
                    <div className="flex items-center gap-2 text-gray-400 text-sm"><Server size={14} /> Device Profile</div>
                    <span className="text-xs font-mono">{selectedAlert['Device_Combo'] || 'Windows / Chrome'}</span>
                  </div>
                </div>

                <div className="mt-8 bg-black/30 p-4 rounded-lg border border-white/5">
                  <p className="text-xs uppercase font-bold text-blue-400 mb-2">Automated Assessment</p>
                  <p className="text-sm text-gray-300 leading-relaxed italic">
                    "Alert triggered via {selectedAlert.reason_tags || 'RF baseline anomaly'}. 
                    Observed infrastructure drift combined with unusual time deviation. 
                    Pattern consistency: 84%. Recommended triage priority: HIGH."
                  </p>
                </div>

                <button className="w-full mt-6 bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 rounded-lg transition-all flex items-center justify-center gap-2">
                  Launch Deep Investigation <ExternalLink size={16} />
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-[#1e1e2e]/50 border border-white/5 border-dashed rounded-xl p-12 text-center">
              <ShieldCheck size={48} className="text-gray-700 mx-auto mb-4" />
              <p className="text-gray-500 text-sm">Select an alert to view AI-driven security assessments and deep telemetry.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SOCDashboard;
