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
  Server,
  Zap,
  TrendingUp,
  Globe,
  Database,
  BarChart3,
  Terminal,
  FileText,
  MousePointer2,
  PieChart,
  Layers,
  PlayCircle,
  FastForward,
  RotateCcw
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell,
  AreaChart,
  Area
} from 'recharts';
import { 
  MapContainer, 
  TileLayer, 
  Marker, 
  Popup, 
  useMap,
  CircleMarker
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default marker icons in Leaflet + Vite
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const API_BASE = "http://localhost:8000/api";

const SOCDashboard = () => {
  const [stats, setStats] = useState(null);
  const [critical, setCritical] = useState([]);
  const [suspicious, setSuspicious] = useState([]);
  const [importance, setImportance] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [activeView, setActiveView] = useState('monitors');
  const [activeTab, setActiveTab] = useState('critical');
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [investigation, setInvestigation] = useState(null);
  const [investigating, setInvestigating] = useState(false);
  const [center, setCenter] = useState([20, 0]);
  const [zoom, setZoom] = useState(2);
  const [alertStatuses, setAlertStatuses] = useState({}); // Tracking { alertId: 'Resolved' | 'In Progress' }
  const [chatLog, setChatLog] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [asnDist, setAsnDist] = useState([]);
  const [geoDist, setGeoDist] = useState([]);
  const [riskyUsers, setRiskyUsers] = useState([]);
  const [scoreDist, setScoreDist] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [alertFilter, setAlertFilter] = useState("");
  const [replayStep, setReplayStep] = useState(0);
  const [showScenario, setShowScenario] = useState(false);



  useEffect(() => {
    fetchData();
  }, []);


  const flyTo = (lat, lng, z = 10) => {
    setCenter([lat, lng]);
    setZoom(z);
  };


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

      const respTimeline = await fetch(`${API_BASE}/graphs/risk-timeline`);
      setTimeline(await respTimeline.json());

      const resAsn = await fetch(`${API_BASE}/graphs/asn-distribution`);
      setAsnDist(await resAsn.json());

      const resGeo = await fetch(`${API_BASE}/graphs/country-distribution`);
      setGeoDist(await resGeo.json());

      const resUsers = await fetch(`${API_BASE}/graphs/top-risky-users`);
      setRiskyUsers(await resUsers.json());

      const resScore = await fetch(`${API_BASE}/graphs/score-distribution`);
      setScoreDist(await resScore.json());

      const resClusters = await fetch(`${API_BASE}/graphs/attack-clusters`);
      setClusters(await resClusters.json());
    } catch (e) {
      console.error("Fetch error:", e);
    }
  };

  const runInvestigation = async (user_id) => {
    setInvestigating(true);
    try {
      const resp = await fetch(`${API_BASE}/investigate?user_id=${user_id}`);
      const data = await resp.json();
      setInvestigation(data.report);
    } catch (e) {
      setInvestigation("Critical failure in Agentic Layer. Check API logs.");
    } finally {
      setInvestigating(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (selectedAlert) {
      setInvestigation(null);
      runInvestigation(selectedAlert['User_ID']);
    }
  }, [selectedAlert]);


  const StatBox = ({ title, value, label, icon: Icon, colorClass }) => (
    <div className="bg-[#161623] border border-white/5 rounded-2xl p-6 relative overflow-hidden group">
      <div className={`absolute top-0 right-0 w-32 h-32 blur-3xl opacity-10 rounded-full bg-${colorClass}-500 -mr-16 -mt-16 group-hover:opacity-20 transition-opacity`}></div>
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-xl bg-white/5 border border-white/10 text-${colorClass}-400`}>
          <Icon size={20} />
        </div>
      </div>
      <h3 className="text-3xl font-black text-white tracking-tight">{value}</h3>
      <p className="text-xs font-bold text-gray-500 uppercase tracking-widest mt-1">{title}</p>
      <p className="text-[10px] text-gray-600 font-mono mt-2 tracking-tighter uppercase">{label}</p>
    </div>
  );

  const renderMonitors = () => (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in slide-in-from-bottom-6 duration-700">
      <div className="lg:col-span-8 space-y-8">
        {/* Search & Filter Bar */}
        <div className="bg-[#111814] border border-emerald-500/10 rounded-[32px] p-4 flex items-center gap-4">
           <div className="relative flex-1">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
              <input 
                type="text"
                value={alertFilter}
                onChange={(e) => setAlertFilter(e.target.value)}
                placeholder="Filter by ASN, IP, or User ID..."
                className="w-full bg-white/5 border-none focus:ring-0 rounded-2xl py-3 pl-12 pr-4 text-xs text-white"
              />
           </div>
           {alertFilter && (
             <div className="flex items-center gap-3">
               <div className="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
                  <span className="text-[9px] font-black text-emerald-500 uppercase">Analysis Active: {alertFilter}</span>
               </div>
               <button 
                 onClick={() => setAlertFilter("")}
                 className="text-[9px] font-black uppercase text-gray-500 hover:text-white"
               >
                 [Reset]
               </button>
             </div>
           )}
        </div>

        <div className="bg-[#111814] border border-emerald-500/10 rounded-[40px] overflow-hidden shadow-2xl">
          <div className="p-8 border-b border-white/5 flex items-center justify-between">
            <div>
              <h3 className="text-white font-black text-xs uppercase tracking-[0.3em]">Detection Queue</h3>
              <p className="text-[10px] text-gray-500 font-bold mt-1 uppercase">Live Stream • High Confidence</p>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="text-[9px] font-black uppercase tracking-[0.2em] text-gray-600 border-b border-white/5">
                  <th className="px-8 py-6">Identity</th>
                  <th className="px-8 py-6">Risk Index</th>
                  <th className="px-8 py-6">Status</th>
                  <th className="px-8 py-6 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {[...critical, ...suspicious]
                  .filter(a => 
                    !alertFilter || 
                    String(a.User_ID).toLowerCase().includes(alertFilter.toLowerCase()) ||
                    String(a.ASN).toLowerCase().includes(alertFilter.toLowerCase()) ||
                    String(a.Device_Combo).toLowerCase().includes(alertFilter.toLowerCase())
                  )
                  .map((alert, i) => (
                    <tr 
                      key={i} 
                      onClick={() => setSelectedAlert(alert)}
                      className="hover:bg-white/[0.02] cursor-pointer group transition-all"
                    >
                      <td className="px-8 py-6">
                        <div className="text-xs font-black text-white">UID_{String(alert.User_ID).slice(-5)}</div>
                        <div className="text-[9px] text-gray-500 font-mono">{alert.ASN || 'Unknown ASN'}</div>
                      </td>
                      <td className="px-8 py-6">
                        <div className="w-24 h-1 bg-white/5 rounded-full overflow-hidden">
                          <div className="h-full bg-emerald-500" style={{ width: `${(alert.final_score || 0) * 100}%` }}></div>
                        </div>
                      </td>
                      <td className="px-8 py-6">
                        <span className="px-2 py-1 rounded-md bg-emerald-500/10 text-[9px] font-black text-emerald-500 uppercase">Active</span>
                      </td>
                      <td className="px-8 py-6 text-right">
                        <ChevronRight size={14} className="text-gray-500 group-hover:text-white inline" />
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Activity Graph */}
        <div className="bg-[#111814] border border-emerald-500/10 rounded-3xl p-8 shadow-2xl relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/5 blur-[100px] rounded-full -mr-32 -mt-32"></div>
          <div className="flex items-center justify-between mb-8">
            <div>
              <h3 className="text-lg font-black text-white flex items-center gap-3">
                <Activity size={20} className="text-emerald-400" />
                STABILITY & RISK TIMELINE
              </h3>
              <p className="text-xs text-gray-500 font-medium mt-1 uppercase tracking-wider italic">Historical Trend Analysis (24h Window)</p>
            </div>
            <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_#10b981]"></div>
          </div>
          <div style={{ height: '320px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timeline}>
                <defs>
                  <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#ffffff03" />
                <XAxis dataKey="hour" axisLine={false} tickLine={false} tick={{fill: '#4b5563', fontSize: 10}} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#4b5563', fontSize: 10}} />
                <Tooltip contentStyle={{ backgroundColor: '#0a0f0d', borderRadius: '12px', border: '1px solid #10b98120', fontSize: '10px', fontWeight: 'bold' }} />
                <Area type="monotone" dataKey="alerts" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorRisk)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Alerts Table Layer */}
        <div className="bg-[#111814] border border-emerald-500/10 rounded-3xl overflow-hidden shadow-2xl">
          <div className="flex px-8 border-b border-white/5 bg-emerald-500/[0.02]">
            <button 
              onClick={() => setActiveTab('critical')}
              className={`pb-5 pt-6 text-[11px] font-black uppercase tracking-[0.2em] transition-all mr-10 flex items-center gap-2 ${activeTab === 'critical' ? 'text-red-500 border-b-2 border-red-500' : 'text-gray-500 hover:text-emerald-500'}`}
            >
              <ShieldAlert size={14} /> Tier 1 Critical ({critical.length})
            </button>
            <button 
              onClick={() => setActiveTab('suspicious')}
              className={`pb-5 pt-6 text-[11px] font-black uppercase tracking-[0.2em] transition-all flex items-center gap-2 ${activeTab === 'suspicious' ? 'text-emerald-500 border-b-2 border-emerald-500' : 'text-gray-500 hover:text-emerald-500'}`}
            >
              <Search size={14} /> Tier 2 Suspicious
            </button>
          </div>
          
          <div className="overflow-x-auto min-h-[500px]">
            <table className="w-full text-left">
              <thead className="text-gray-600 text-[10px] uppercase font-black tracking-widest border-b border-white/5">
                <tr>
                  <th className="px-8 py-5">Event ID</th>
                  <th className="px-8 py-5">User</th>
                  <th className="px-8 py-5">Risk Matrix</th>
                  <th className="px-8 py-5">Status</th>
                  <th className="px-8 py-5">Signatures</th>
                  <th className="px-8 py-5 text-right">Triage</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {(activeTab === 'critical' ? critical : suspicious).slice(0, 30).map((alert, idx) => (
                  <tr 
                    key={idx} 
                    onClick={() => {
                      console.log("Selecting alert:", alert.alert_rank);
                      setSelectedAlert(alert);
                    }}
                    className={`hover:bg-white/[0.02] cursor-pointer group transition-all ${selectedAlert?.alert_rank === alert.alert_rank ? 'bg-blue-500/[0.05]' : ''}`}
                  >
                    <td className="px-8 py-6 font-mono text-[10px] text-gray-500 group-hover:text-blue-400 italic">EVNT_{1000 + alert.alert_rank}</td>
                    <td className="px-8 py-6 uppercase font-black text-xs text-white tracking-widest">UID_{String(alert['User_ID']).slice(-5)}</td>
                    <td className="px-8 py-6">
                      <div className="flex items-center gap-3">
                        <div className="w-16 h-1 bg-white/5 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${activeTab === 'critical' ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]' : 'bg-blue-500'}`} 
                            style={{ width: `${(alert.final_score || alert.risk_score) * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-[10px] font-black font-mono text-gray-400">
                          {Math.round((alert.final_score || alert.risk_score) * 100)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-8 py-6">
                       <span className={`px-2 py-0.5 rounded-full text-[8px] font-black uppercase tracking-widest border ${
                         alertStatuses[alert.alert_rank] === 'Resolved' ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400' :
                         alertStatuses[alert.alert_rank] === 'In Progress' ? 'bg-amber-500/20 border-amber-500/40 text-amber-400' :
                         'bg-white/5 border-white/10 text-gray-500'
                       }`}>
                         {alertStatuses[alert.alert_rank] || 'New Entry'}
                       </span>
                    </td>
                    <td className="px-8 py-6">
                      <div className="flex flex-wrap gap-1.5">
                        {(alert.reason_tags || "RF_DETECTED").split(';').filter(t => t.trim()).slice(0, 2).map((tag, i) => (
                          <span key={i} className={`px-2 py-0.5 rounded-md border text-[8px] font-black uppercase tracking-wider ${activeTab === 'critical' ? 'bg-red-500/10 border-red-500/20 text-red-400' : 'bg-blue-500/10 border-blue-500/20 text-blue-400'}`}>
                            {tag.trim()}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="px-8 py-6 text-right">
                      <ChevronRight size={14} className="text-gray-500 group-hover:text-blue-400 transition-transform group-hover:translate-x-1 inline" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="lg:col-span-4 space-y-8">
        {selectedAlert ? (
          <div className="bg-[#111814] border border-emerald-500/20 rounded-[32px] overflow-hidden shadow-2xl shadow-emerald-500/5 animate-in slide-in-from-right-8 duration-700 flex flex-col h-full max-h-[85vh]">
            {/* Dossier Header */}
            <div className="p-6 bg-gradient-to-br from-emerald-600/20 to-teal-900/40 flex items-center justify-between border-b border-white/5">
              <div className="flex items-center gap-3">
                <div className="px-2 py-0.5 bg-red-500 text-white text-[8px] font-black uppercase rounded">Tier 1</div>
                <span className="text-[10px] font-black uppercase tracking-[0.2em] text-emerald-400">Tactical Dossier</span>
              </div>
              <div className={`px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-widest border ${
                  alertStatuses[selectedAlert.alert_rank] === 'Resolved' ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400' :
                  alertStatuses[selectedAlert.alert_rank] === 'In Progress' ? 'bg-amber-500/20 border-amber-500/40 text-amber-400' :
                  'bg-white/5 border-white/20 text-gray-400'
              }`}>
                {alertStatuses[selectedAlert.alert_rank] || 'New Case'}
              </div>
            </div>
            
            <div className="flex-1 overflow-y-auto p-8 space-y-10 custom-scrollbar">
              {/* SECTION: 📌 Summary */}
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <ShieldAlert size={18} className="text-emerald-500" />
                  <h3 className="text-xs font-black uppercase tracking-widest text-white">Subject Identification</h3>
                </div>
                <div className="bg-white/5 border border-white/5 rounded-3xl p-6 relative group">
                  <div className="absolute top-4 right-4 text-[24px] font-black text-white/5 group-hover:text-emerald-500/10 transition-colors">#{selectedAlert.alert_rank}</div>
                  <div className="mb-4">
                    <p className="text-[9px] font-black text-gray-500 uppercase tracking-widest mb-1">Authenticated UID</p>
                    <p className="text-lg font-black text-white break-all">{selectedAlert.User_ID}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                     <div>
                        <p className="text-[9px] font-black text-gray-500 uppercase tracking-widest mb-1">Risk Score</p>
                        <p className={`text-xl font-black ${selectedAlert.final_score > 0.5 ? 'text-red-500' : 'text-emerald-400'}`}>{Math.round(selectedAlert.final_score * 100)}%</p>
                     </div>
                     <div>
                        <p className="text-[9px] font-black text-gray-500 uppercase tracking-widest mb-1">Alert Category</p>
                        <p className="text-xs font-bold text-white uppercase italic">{selectedAlert.alert_tier || 'Critical'}</p>
                     </div>
                  </div>
                </div>
              </div>

              {/* SECTION: 🧠 AI Explanation */}
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <Zap size={18} className="text-emerald-400" />
                  <h3 className="text-xs font-black uppercase tracking-widest text-white">Agentic Reasoning</h3>
                </div>
                <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-3xl p-6 relative">
                  {investigating ? (
                    <div className="flex flex-col items-center gap-4 py-8">
                       <Activity size={32} className="text-emerald-500 animate-pulse" />
                       <p className="text-[10px] uppercase font-black tracking-[0.2em] text-emerald-500/60 animate-pulse italic">Thinking...</p>
                    </div>
                  ) : (
                    <p className="text-[11px] text-emerald-100/80 leading-relaxed font-medium">
                      {investigation || `Detection triggered by ${selectedAlert.reason_tags}. Initial indicators suggest a high-confidence behavioral drift from historical norms.`}
                    </p>
                  )}
                </div>
              </div>

              {/* SECTION: 📊 Behavioral Breakdown */}
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <TrendingUp size={18} className="text-blue-400" />
                  <h3 className="text-xs font-black uppercase tracking-widest text-white">Behavioral Anomaly Engine</h3>
                </div>
                <div className="grid grid-cols-1 gap-4">
                   <div className="bg-white/5 border border-white/5 rounded-2xl p-4 flex items-center justify-between">
                      <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Login Frequency</span>
                      <span className="text-xs font-black text-white">{selectedAlert.User_Login_Count_Prior ? Math.round(selectedAlert.User_Login_Count_Prior) : 12} Events</span>
                   </div>
                   <div className="bg-white/5 border border-white/5 rounded-2xl p-4 flex items-center justify-between">
                      <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Failure Pressure (1h)</span>
                      <span className={`text-xs font-black ${selectedAlert.Fail_Count_1Hour > 2 ? 'text-red-500' : 'text-emerald-500'}`}>{selectedAlert.Fail_Count_1Hour || 0} Bursts</span>
                   </div>
                   <div className="bg-white/5 border border-white/5 rounded-2xl p-4 flex items-center justify-between">
                      <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Hour Deviation</span>
                      <span className="text-xs font-black text-white">{Math.round((selectedAlert.User_Login_Hour_Deviation || 0) * 100)}% Sigma</span>
                   </div>
                </div>
              </div>

              {/* SECTION: 🌐 Infrastructure Context */}
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <Globe size={18} className="text-purple-400" />
                  <h3 className="text-xs font-black uppercase tracking-widest text-white">Network & Geo Intelligence</h3>
                </div>
                <div className="space-y-4">
                  <div className="bg-white/5 border border-white/5 rounded-2xl p-5">
                    <div className="flex items-center justify-between mb-4">
                       <span className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Infrastructure Path</span>
                       <span className="text-[10px] font-black text-emerald-500">AS{selectedAlert.ASN} • {selectedAlert.Country}</span>
                    </div>
                    <div className="bg-black/40 rounded-xl p-3 border border-white/5 text-[10px] font-mono text-gray-400 truncate">
                      OS: {String(selectedAlert.Device_Combo).split('-')[1] || 'Unknown'} / V_{Math.round((selectedAlert.Subnet_Change_Rate_Prior || 0.1)*100)}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white/5 border border-white/5 rounded-2xl p-4 text-center">
                       <p className="text-[9px] font-black text-gray-500 uppercase tracking-widest mb-1">ASN Drift</p>
                       <p className="text-xs font-black text-white">{(selectedAlert.ASN_Change_Rate_Prior || 0.05).toFixed(2)}</p>
                    </div>
                    <div className="bg-white/5 border border-white/5 rounded-2xl p-4 text-center">
                       <p className="text-[9px] font-black text-gray-500 uppercase tracking-widest mb-1">Geo Vol</p>
                       <p className="text-xs font-black text-white">{(selectedAlert.Country_Change_Rate_Prior || 0.02).toFixed(2)}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* SECTION: ⏱ Timeline View */}
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <Clock size={18} className="text-teal-400" />
                  <h3 className="text-xs font-black uppercase tracking-widest text-white">Tactical Event Trace</h3>
                </div>
                <div className="space-y-6 border-l-2 border-emerald-500/10 ml-2 pl-6">
                   <div className="relative">
                      <div className="absolute -left-[31px] top-1 w-2 h-2 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]"></div>
                      <p className="text-[10px] font-black text-white uppercase italic">Active Trigger: {selectedAlert.Login_Timestamp}</p>
                      <p className="text-[9px] text-gray-500 mt-1 uppercase">Node Entry via {selectedAlert['IP Address']}</p>
                   </div>
                   <div className="relative">
                      <div className="absolute -left-[31px] top-1 w-2 h-2 rounded-full bg-emerald-500/20"></div>
                      <p className="text-[10px] font-black text-gray-500 uppercase italic">Prior Session: 24h Window</p>
                      <p className="text-[9px] text-gray-600 mt-1 uppercase">Successful auth from known ASN</p>
                   </div>
                </div>
              </div>
            </div>

            {/* Tactical Triage Footer */}
            <div className="p-8 bg-black/40 border-t border-white/5 grid grid-cols-3 gap-4">
                <button 
                  onClick={() => setAlertStatuses({...alertStatuses, [selectedAlert.alert_rank]: 'Resolved'})}
                  className="bg-emerald-600/10 hover:bg-emerald-600 border border-emerald-600/30 text-emerald-400 hover:text-white font-black text-[10px] uppercase tracking-widest py-4 rounded-2xl transition-all shadow-lg hover:shadow-emerald-500/20"
                >
                  Resolve
                </button>
                <button 
                  onClick={() => setAlertStatuses({...alertStatuses, [selectedAlert.alert_rank]: 'In Progress'})}
                  className="bg-amber-600/10 hover:bg-amber-600 border border-amber-600/30 text-amber-400 hover:text-white font-black text-[10px] uppercase tracking-widest py-4 rounded-2xl transition-all shadow-lg hover:shadow-amber-500/20"
                >
                  Escalate
                </button>
                <button 
                  onClick={() => setAlertStatuses({...alertStatuses, [selectedAlert.alert_rank]: 'Closed'})}
                  className="bg-white/5 hover:bg-white/10 border border-white/10 text-gray-500 font-black text-[10px] uppercase tracking-widest py-4 rounded-2xl transition-all"
                >
                  Ignore
                </button>
            </div>
          </div>
        ) : (
          <div className="bg-[#111814] border border-white/5 border-dashed rounded-[40px] p-20 text-center flex flex-col items-center justify-center min-h-[500px]">
             <ShieldCheck size={48} className="text-emerald-950 mb-6" />
             <p className="text-xs font-black uppercase text-emerald-900 tracking-widest">Awaiting Selection</p>
          </div>
        )}
      </div>
    </div>
  );

  const renderThreatIntel = () => (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div className="lg:col-span-4 bg-[#111814] border border-emerald-500/10 rounded-3xl p-8 shadow-2xl">
        <h3 className="text-emerald-400 font-black text-xs uppercase tracking-widest mb-6 flex items-center gap-2">
          <Globe size={16} /> Global IP Reputation Hotspots
        </h3>
        <div className="space-y-4">
          {[
            { ip: '45.128.232.14', risk: '98%', geo: 'RU/Moscow', tags: 'Credential Stuffing', lat: 55.75, lng: 37.61 },
            { ip: '185.220.101.44', risk: '92%', geo: 'DE/Tor-Node', tags: 'Anomaly-Burst', lat: 52.52, lng: 13.40 },
            { ip: '103.14.26.11', risk: '84%', geo: 'CN/Beijing', tags: 'Brute Force', lat: 39.90, lng: 116.40 },
            { ip: '194.36.191.2', risk: '76%', geo: 'NL/Amsterdam', tags: 'Botnet-Scale', lat: 52.36, lng: 4.90 }
          ].map((item, i) => (
            <div 
              key={i} 
              onClick={() => flyTo(item.lat, item.lng, 10)}
              className="flex items-center justify-between p-4 bg-white/5 rounded-2xl border border-white/5 group hover:border-emerald-500/20 transition-all cursor-pointer"
            >
              <div className="flex items-center gap-4">
                <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]"></div>
                <div>
                  <p className="text-[10px] font-mono font-bold text-emerald-400">{item.ip}</p>
                  <p className="text-[8px] text-gray-500 uppercase font-black">{item.geo}</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-[10px] font-black text-white">{item.risk}</p>
              </div>
            </div>
          ))}
          <button 
            onClick={() => flyTo(20, 0, 2)}
            className="w-full mt-4 py-3 border border-emerald-500/20 rounded-xl text-[9px] font-black uppercase tracking-widest text-emerald-500/60 hover:text-emerald-400 hover:bg-emerald-500/5 transition-all"
          >
            Reset World View
          </button>
        </div>
      </div>

      <div className="lg:col-span-8 bg-[#060a08] border border-emerald-500/10 rounded-3xl overflow-hidden relative group shadow-2xl shadow-emerald-500/5 min-h-[500px]">
        <CyberMap center={center} zoom={zoom} />
        
        <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-[#060a08] via-transparent to-transparent z-10"></div>
        
        <div className="absolute bottom-8 left-8 z-30 pointer-events-none">
          <div className="bg-black/80 backdrop-blur-md border border-emerald-500/20 rounded-2xl p-6">
             <div className="flex items-center gap-4 mb-2">
                <Zap size={20} className="text-emerald-400 animate-pulse" />
                <h4 className="text-white font-black text-sm uppercase tracking-widest">Global Intelligence Grid</h4>
             </div>
             <p className="text-[10px] text-gray-500 font-bold uppercase tracking-widest leading-loose">
               No-Key Satellite Access • <span className="text-emerald-500">Voyager Tier</span> • {center[0].toFixed(2)}, {center[1].toFixed(2)}
             </p>
          </div>
        </div>
      </div>
    </div>
  );


  const handleAgentChat = () => {
    if (!chatInput.trim()) return;
    const msg = chatInput;
    setChatLog([...chatLog, { role: 'user', text: msg }]);
    setChatInput("");
    
    // Auto-response simulation
    setTimeout(() => {
      setChatLog(prev => [...prev, { 
        role: 'ai', 
        text: `LOG_ENTRY: Analysing query context for "${msg}". Scanning global ASN threat database... Correlation complete. High-risk behavioral markers identified in the subnet range. See Dossier for details.` 
      }]);
    }, 1000);
  };

  const replayScenario = [
    { title: "Baseline Baseline", time: "10:00 AM", score: 0.02, detail: "User logged in from habitual UK ASN. Device fingerprint matched history perfectly.", status: "SAFE" },
    { title: "Network Recon", time: "11:45 AM", score: 0.18, detail: "Password recovery attempt detected from AWS Proxy node in US East. Minor feature drift.", status: "SUSPICIOUS" },
    { title: "Behavioral Pivot", time: "12:12 PM", score: 0.45, detail: "Successful login from a previously unseen Android version. Geographic velocity: 540mph.", status: "ELEVATED" },
    { title: "Compromise Trigger", time: "12:13 PM", score: 0.92, detail: "Immediate lateral movement attempt on sensitive API. 5 failed bursts in 6 seconds. ALERT FIRED.", status: "CRITICAL" }
  ];

  const renderReplay = () => (
    <div className="max-w-6xl mx-auto space-y-12 animate-in fade-in duration-800">
       <div className="flex items-center justify-between">
          <div>
             <h2 className="text-2xl font-black text-white uppercase tracking-tighter">Case Study: Anatomy of an ATO</h2>
             <p className="text-[10px] text-emerald-500 font-bold uppercase tracking-widest mt-1 italic">Scenario ID: #ALPHA-BRAVO-713 • Historical Replay Mode</p>
          </div>
          <button 
             onClick={() => { setReplayStep(0); setShowScenario(true); }}
             className="px-8 py-3 bg-emerald-600 rounded-2xl text-white font-black text-[10px] uppercase tracking-widest hover:bg-emerald-500 transition-all flex items-center gap-3 shadow-lg shadow-emerald-600/20"
          >
             <PlayCircle size={16} /> Start Case Replay
          </button>
       </div>

       {showScenario && (
         <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
            {/* Timeline View */}
            <div className="lg:col-span-4 bg-[#111814] border border-white/5 rounded-[40px] p-10 space-y-8">
               <h3 className="text-[10px] font-black uppercase text-gray-500 tracking-[0.3em] mb-4">Event Sequence</h3>
               <div className="space-y-10 border-l border-white/5 ml-4 pl-10">
                  {replayScenario.map((s, i) => (
                    <div 
                      key={i} 
                      className={`relative transition-all duration-500 ${replayStep >= i ? 'opacity-100 translate-x-0' : 'opacity-20 translate-x-4'}`}
                    >
                       <div className={`absolute -left-[53px] top-1 w-6 h-6 rounded-full border-4 border-[#090909] ${replayStep === i ? 'bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)] scale-125' : replayStep > i ? 'bg-gray-700' : 'bg-white/5'} transition-all`}></div>
                       <p className="text-[8px] font-black text-gray-500 uppercase tracking-widest mb-1">{s.time}</p>
                       <h4 className={`text-sm font-black uppercase mb-1 ${replayStep === i ? 'text-white' : 'text-gray-600'}`}>{s.title}</h4>
                       {replayStep === i && <p className="text-[10px] text-emerald-100/60 leading-relaxed italic">{s.detail}</p>}
                    </div>
                  ))}
               </div>
            </div>

            {/* Visual Intelligence Section */}
            <div className="lg:col-span-8 space-y-10">
               <div className="bg-[#111814] border border-emerald-500/10 rounded-[40px] p-10 h-[400px]">
                  <h3 className="text-[10px] font-black uppercase text-blue-400 tracking-[0.3em] mb-8">Risk Progression Telemetry</h3>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={replayScenario.slice(0, replayStep + 1)}>
                       <XAxis dataKey="time" hide />
                       <YAxis domain={[0, 1]} hide />
                       <Tooltip labelStyle={{ display: 'none' }} contentStyle={{ background: '#0a0f0d', border: '1px solid #10b98120', fontSize: '10px' }} />
                       <Area type="monotone" dataKey="score" stroke="#10b981" fill="url(#replayGradient)" strokeWidth={4} strokeLinecap="round" />
                       <defs>
                          <linearGradient id="replayGradient" x1="0" y1="0" x2="0" y2="1">
                             <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                             <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                          </linearGradient>
                       </defs>
                    </AreaChart>
                  </ResponsiveContainer>
               </div>

               <div className="bg-black/40 border border-white/5 rounded-[40px] p-10 flex items-center justify-between">
                  <div>
                    <span className="text-[10px] font-black text-gray-500 uppercase tracking-widest block mb-4">Tactical Status</span>
                    <div className="flex items-center gap-4">
                       <span className={`px-6 py-2 rounded-2xl text-xs font-black uppercase tracking-widest border transition-all ${
                         replayScenario[replayStep].status === 'SAFE' ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-500' :
                         replayScenario[replayStep].status === 'SUSPICIOUS' ? 'bg-blue-500/10 border-blue-500/20 text-blue-500' :
                         'bg-red-500/20 border-red-500/40 text-red-500 animate-pulse'
                       }`}>
                         {replayScenario[replayStep].status}
                       </span>
                       <div className="w-1.5 h-1.5 rounded-full bg-gray-500"></div>
                       <span className="text-[10px] font-black text-white italic">SCORE: {Math.round(replayScenario[replayStep].score * 100)}% Saturation</span>
                    </div>
                  </div>
                  
                  <div className="flex gap-4">
                     <button 
                       disabled={replayStep === 0}
                       onClick={() => setReplayStep(Math.max(0, replayStep - 1))}
                       className="p-4 bg-white/5 rounded-2xl hover:bg-white/10 disabled:opacity-20 transition-all"
                     >
                       <RotateCcw size={20} className="text-gray-400" />
                     </button>
                     <button 
                       disabled={replayStep === replayScenario.length - 1}
                       onClick={() => setReplayStep(Math.min(replayScenario.length - 1, replayStep + 1))}
                       className="px-10 py-4 bg-emerald-600 rounded-2xl text-white font-black text-[11px] uppercase tracking-[0.2em] flex items-center gap-3 hover:bg-emerald-500 disabled:opacity-20 transition-all shadow-xl shadow-emerald-500/20"
                     >
                        Forward <FastForward size={16} />
                     </button>
                  </div>
               </div>
            </div>
         </div>
       )}

       {!showScenario && (
          <div className="py-40 bg-[#111814] border border-white/5 border-dashed rounded-[40px] flex flex-col items-center justify-center text-center">
             <div className="p-8 bg-emerald-500/10 rounded-full mb-8">
                <PlayCircle size={48} className="text-emerald-500" />
             </div>
             <h3 className="text-xl font-black text-white uppercase mb-2">Scenario Library Ready</h3>
             <p className="text-xs text-gray-500 font-medium max-w-sm leading-relaxed">
                Step through recorded Account Takeover events to evaluate model precision and defense orchestration.
             </p>
          </div>
       )}
    </div>
  );

  const renderClusters = () => (
    <div className="space-y-12 animate-in fade-in zoom-in-95 duration-700">
       <div className="flex items-center justify-between mb-4">
          <div>
             <h2 className="text-xl font-black text-white uppercase tracking-tighter">Campaign Correlation Engine</h2>
             <p className="text-[10px] text-emerald-500 font-bold uppercase tracking-widest mt-1 italic">Active Pattern Matcher • Multi-User Vectors Identified</p>
          </div>
       </div>

       <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {(clusters || []).length > 0 ? (clusters || []).map((c, i) => (
             <div 
               key={i} 
               onClick={() => {
                 // Tactical pivot: Filter by the core value (ASN or Device)
                 const val = c.value.split(' ')[0]; // Take just the ID/ASN part
                 setAlertFilter(val);
                 setActiveView('monitors');
                 
                 // Feedback toast (internal state check)
                 console.log(`PIVOT_ACTIVE: Targeting ${val} campaign metrics.`);
               }}
               className="bg-[#111814] border border-emerald-500/10 rounded-[32px] p-8 group hover:border-emerald-500/30 transition-all relative overflow-hidden cursor-pointer"
             >
                <div className={`absolute top-0 right-0 w-32 h-32 blur-3xl opacity-20 bg-${c.severity === 'CRITICAL' ? 'red' : 'amber'}-500 -mr-16 -mt-16`}></div>
                
                <div className="flex items-center gap-4 mb-6">
                   <div className={`p-3 rounded-xl ${c.type === 'ASN_NETWORK' ? 'bg-purple-500/20 text-purple-400' : 'bg-blue-500/20 text-blue-400'}`}>
                      {c.type === 'ASN_NETWORK' ? <Globe size={20} /> : <Cpu size={20} />}
                   </div>
                   <div>
                      <span className="text-[8px] font-black text-gray-500 uppercase tracking-widest block mb-1">{c.type.replace('_', ' ')}</span>
                      <h4 className="text-sm font-black text-white">{c.value}</h4>
                   </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-8">
                   <div className="bg-black/40 p-4 rounded-2xl border border-white/5">
                      <p className="text-[9px] font-black text-gray-500 uppercase mb-1">Impact</p>
                      <p className="text-xs font-black text-white">{c.users} SUBJECTS</p>
                   </div>
                   <div className="bg-black/40 p-4 rounded-2xl border border-white/5">
                      <p className="text-[9px] font-black text-gray-500 uppercase mb-1">Signals</p>
                      <p className="text-xs font-black text-white">{c.alerts} EVENTS</p>
                   </div>
                </div>

                <div className="flex items-center justify-between pt-6 border-t border-white/5">
                   <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${c.severity === 'CRITICAL' ? 'bg-red-500 animate-pulse' : 'bg-amber-500'}`}></div>
                      <span className="text-[9px] font-black text-gray-400 uppercase tracking-widest">{c.severity} THREAT</span>
                   </div>
                   <span className="text-[10px] font-black text-emerald-500 uppercase italic">Investigate →</span>
                </div>
             </div>
          )) : (
            <div className="lg:col-span-3 py-20 bg-[#111814] border border-white/5 border-dashed rounded-[40px] flex flex-col items-center justify-center">
               <Layers size={48} className="text-emerald-950 mb-6" />
               <p className="text-xs font-black uppercase text-emerald-900 tracking-widest">No Active Campaigns Detected</p>
            </div>
          )}
       </div>
    </div>
  );

  const renderIntelligence = () => (
    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-8 duration-1000">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
         {/* ASN RISK Heatmap */}
         <div className="bg-[#111814] border border-emerald-500/10 rounded-[40px] p-10 shadow-2xl relative overflow-hidden">
            <h3 className="text-xs font-black uppercase text-emerald-400 tracking-[0.3em] mb-8 flex items-center gap-3">
              <Activity size={16} /> 🔥 Critical ASN Attack Vector
            </h3>
            <div style={{ height: '300px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={asnDist || []} layout="vertical">
                  <XAxis type="number" hide />
                  <YAxis dataKey="asn" type="category" width={80} tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#0a0f0d', border: '1px solid #10b98120', fontSize: '10px' }} cursor={{ fill: '#ffffff05' }} />
                  <Bar dataKey="risk" fill="#10b981" radius={[0, 4, 4, 0]} barSize={20} />
                </BarChart>
              </ResponsiveContainer>
            </div>
         </div>

         {/* Score Saturation */}
         <div className="bg-[#111814] border border-emerald-500/10 rounded-[40px] p-10 shadow-2xl">
            <h3 className="text-xs font-black uppercase text-blue-400 tracking-[0.3em] mb-8 flex items-center gap-3">
              <TrendingUp size={16} /> 📈 Risk Score Saturation
            </h3>
            <div style={{ height: '300px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={scoreDist || []}>
                  <XAxis dataKey="range" tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} />
                  <Tooltip contentStyle={{ background: '#0a0f0d', border: '1px solid #10b98120', fontSize: '10px' }} />
                  <Area type="monotone" dataKey="count" stroke="#3b82f6" fillOpacity={0.3} fill="#3b82f6" strokeWidth={3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
         </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
         {/* Top Risky Countries */}
         <div className="bg-[#111814] border border-emerald-500/10 rounded-[40px] p-10 shadow-2xl">
            <h3 className="text-[10px] font-black uppercase text-purple-400 tracking-[0.3em] mb-8">🌍 Global Anomaly Density</h3>
            <div className="space-y-5">
              {(geoDist || []).slice(0, 5).map((item, i) => (
                <div key={i} className="flex items-center justify-between">
                   <span className="text-xs font-bold text-gray-500">{item.country} NODE</span>
                   <div className="flex-1 mx-4 h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full bg-purple-500/40" style={{ width: geoDist[0]?.count ? `${(item.count / geoDist[0].count) * 100}%` : '0%' }}></div>
                   </div>
                   <span className="text-[10px] font-black text-white">{item.count}</span>
                </div>
              ))}
            </div>
         </div>

         {/* High Risk Subjects */}
         <div className="lg:col-span-2 bg-[#111814] border border-emerald-500/10 rounded-[40px] p-10 shadow-2xl overflow-hidden">
            <h3 className="text-[10px] font-black uppercase text-emerald-400 tracking-[0.3em] mb-8">👤 High-Risk Subject Registry</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="text-[8px] font-black uppercase tracking-widest text-gray-600 border-b border-white/5">
                    <th className="pb-4">Subject UID</th>
                    <th className="pb-4">Mean Risk</th>
                    <th className="pb-4">Anomalies</th>
                    <th className="pb-4">Threat Level</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {(riskyUsers || []).map((u, i) => (
                    <tr key={i} className="group hover:bg-white/[0.02]">
                      <td className="py-4 text-[10px] font-mono text-gray-400">UID_{u.user}</td>
                      <td className="py-4 text-[11px] font-black text-white">{Math.round(u.risk * 100)}%</td>
                      <td className="py-4 text-[10px] text-emerald-500 font-bold">{u.count} EVENTS</td>
                      <td className="py-4">
                        <span className={`px-2 py-0.5 rounded text-[8px] font-black ${u.risk > 0.6 ? 'bg-red-500/20 text-red-500' : 'bg-amber-500/20 text-amber-500'}`}>
                          {u.risk > 0.6 ? 'CRITICAL_WATCH' : 'ELEVATED'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
         </div>
      </div>
    </div>
  );

  const renderAgent = () => (
    <div className="max-w-4xl mx-auto h-[70vh] flex flex-col bg-[#111814] border border-emerald-500/10 rounded-[40px] overflow-hidden animate-in zoom-in-95 duration-500">
      <div className="p-6 border-b border-white/5 flex items-center justify-between bg-emerald-500/5">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/20 rounded-lg"><Terminal size={18} className="text-emerald-400" /></div>
          <div>
            <h3 className="text-[10px] font-black uppercase tracking-widest text-white">Chrono-Investigator v3.0</h3>
            <p className="text-[8px] text-emerald-500 font-bold uppercase italic">Active Thinking Session • Gemini 1.5 Flash</p>
          </div>
        </div>
        <div className="flex gap-2">
           <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
           <div className="w-2 h-2 rounded-full bg-emerald-500/30"></div>
           <div className="w-2 h-2 rounded-full bg-emerald-500/10"></div>
        </div>
      </div>
      <div className="flex-1 p-8 overflow-y-auto space-y-6 custom-scrollbar">
        <div className="flex gap-4">
          <div className="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center text-[10px] font-black text-white shadow-[0_0_10px_rgba(16,185,129,0.3)]">AI</div>
          <div className="bg-white/5 border border-white/5 rounded-3xl rounded-tl-none p-6 max-w-[80%]">
            <p className="text-xs text-gray-300 leading-relaxed italic">
              "Ready for deep analysis. Select an alert from the Monitors tab to initialize the investigation context. I can correlate ASN failure rates, device fingerprints, and rule triggers to provide a Tier-3 summary."
            </p>
          </div>
        </div>
        
        {chatLog.map((log, i) => (
          <div key={i} className={`flex gap-4 ${log.role === 'user' ? 'flex-row-reverse' : ''}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-black text-white shadow-lg ${log.role === 'user' ? 'bg-blue-600 shadow-blue-500/20' : 'bg-emerald-600 shadow-emerald-500/20'}`}>
              {log.role === 'user' ? 'SOC' : 'AI'}
            </div>
            <div className={`${log.role === 'user' ? 'bg-blue-500/10 border-blue-500/20' : 'bg-white/5 border-white/5'} border rounded-3xl p-6 max-w-[80%]`}>
              <p className={`text-xs leading-relaxed ${log.role === 'user' ? 'text-blue-100' : 'text-gray-300'}`}>
                {log.text}
              </p>
            </div>
          </div>
        ))}

        {investigation && (
           <div className="flex gap-4">
            <div className="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center text-[10px] font-black text-white shadow-[0_0_10px_rgba(16,185,129,0.3)]">AI</div>
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-3xl rounded-tl-none p-6 max-w-[80%]">
              <p className="text-xs text-emerald-100 leading-relaxed font-medium">
                {investigation}
              </p>
            </div>
          </div>
        )}
      </div>
      <div className="p-6 bg-black/20 border-t border-white/5">
         <form onSubmit={(e) => { e.preventDefault(); handleAgentChat(); }} className="relative">
            <input 
              type="text" 
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Ask the Agent about specific users or IP ranges..." 
              className="w-full bg-white/5 border border-white/5 rounded-2xl py-4 px-6 text-xs text-white placeholder:text-gray-600 focus:outline-none focus:border-emerald-500/30 transition-all"
            />
            <button 
              type="submit"
              className="absolute right-4 top-1/2 -translate-y-1/2 p-2 bg-emerald-600 rounded-xl text-white hover:bg-emerald-500 transition-all"
            >
              <ChevronRight size={16} />
            </button>
         </form>
      </div>
    </div>
  );

  const renderReports = () => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 animate-in fade-in duration-700">
      {[
        { name: 'Inference Performance', type: 'JSON', size: '368 B', date: 'Today' },
        { name: 'Critical Alerts Export', type: 'CSV', size: '4.9 KB', date: '1 hour ago' },
        { name: 'Suspicious Activities', type: 'CSV', size: '951 KB', date: '2 hours ago' },
        { name: 'Model Drift Analysis', type: 'JSON', size: '4.2 KB', date: 'Yesterday' }
      ].map((report, i) => (
        <div key={i} className="bg-[#111814] border border-emerald-500/10 rounded-3xl p-6 group hover:scale-[1.02] transition-all cursor-pointer">
          <div className="flex items-center justify-between mb-6">
            <div className="p-3 bg-white/5 rounded-xl text-emerald-400 group-hover:bg-emerald-500 group-hover:text-white transition-all">
              <FileText size={20} />
            </div>
            <span className="text-[8px] font-black text-gray-600 uppercase tracking-widest bg-black/40 px-2 py-1 rounded-md">{report.type}</span>
          </div>
          <h4 className="text-white font-black text-sm uppercase mb-1">{report.name}</h4>
          <p className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">{report.size} • {report.date}</p>
          <div className="mt-6 pt-6 border-t border-white/5 flex items-center justify-between text-[10px] uppercase font-black tracking-widest text-emerald-500 opacity-0 group-hover:opacity-100 transition-all">
            Download <ExternalLink size={12} />
          </div>
        </div>
      ))}
    </div>
  );

  const renderPlaceholder = (title) => (
    <div className="flex flex-col items-center justify-center h-[60vh] bg-[#111814] border border-white/5 border-dashed rounded-[40px] animate-in zoom-in-95 duration-500">
      <Zap size={48} className="text-emerald-950 mb-4 animate-pulse" />
      <h3 className="text-xl font-black text-emerald-900 uppercase tracking-[0.3em]">{title}</h3>
      <p className="text-[10px] text-emerald-900/40 font-bold uppercase mt-2 italic">Module Under Development • SOC v4.0</p>
    </div>
  );


  return (
    <div className="min-h-screen bg-[#060a08] text-gray-200 p-6 lg:p-10 font-sans selection:bg-emerald-500/30 relative overflow-hidden">
      {/* Dynamic Cyber Gradient Background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-emerald-900/10 blur-[120px] rounded-full"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-emerald-900/10 blur-[120px] rounded-full"></div>
        <div className="absolute top-[20%] right-[10%] w-[30%] h-[30%] bg-emerald-500/5 blur-[150px] rounded-full"></div>
      </div>

      {/* Navigation Layer */}
      <nav className="flex flex-col md:flex-row items-center justify-between mb-12 bg-[#111814]/80 backdrop-blur-3xl border border-emerald-500/10 rounded-3xl px-10 py-6 sticky top-0 z-50 shadow-2xl">
        <div className="flex items-center gap-6 mb-4 md:mb-0">
          <div className="bg-gradient-to-br from-emerald-500 to-teal-700 p-3 rounded-2xl shadow-xl shadow-emerald-500/20">
            <ShieldCheck size={24} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-black tracking-[-0.05em] text-white uppercase italic">Chrono-SOC Console</h1>
            <p className="text-[9px] font-black text-emerald-500 uppercase tracking-[0.3em] mt-0.5 underline decoration-emerald-500/50 underline-offset-4">Agentic Intelligence Engine</p>
          </div>
        </div>

        
        <div className="flex bg-black/40 p-1.5 rounded-2xl border border-white/5 mb-4 md:mb-0">
          {[
            { id: 'monitors', icon: BarChart3, label: 'Monitors' },
            { id: 'threats', icon: Zap, label: 'Threat Intel' },
            { id: 'intelligence', icon: PieChart, label: 'Intel Matrix' },
            { id: 'clusters', icon: Layers, label: 'Attack Clusters' },
            { id: 'replay', icon: PlayCircle, label: 'Case Study' },
            { id: 'agent', icon: Terminal, label: 'Analysis Agent' },
            { id: 'reports', icon: FileText, label: 'Reports' }
          ].map(tab => (
            <button 
              key={tab.id}
              onClick={() => setActiveView(tab.id)}
              className={`px-5 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all flex items-center gap-2 ${activeView === tab.id ? 'bg-emerald-500 text-black shadow-[0_0_20px_rgba(16,185,129,0.3)]' : 'text-gray-500 hover:text-gray-300'}`}
            >
              <tab.icon size={14} /> {tab.label}
            </button>
          ))}
        </div>


        <div className="flex items-center gap-5">
           <div className="flex flex-col items-end">
              <span className="text-[10px] font-black text-white italic">SOC_ANALYST_001</span>
              <span className="text-[8px] font-bold text-gray-600 uppercase tracking-widest">Active Terminal</span>
           </div>
           <div className="w-10 h-10 rounded-full border-2 border-blue-500/30 p-0.5 animate-spin-slow">
              <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=Felix" className="w-full h-full rounded-full bg-white/5" alt="soc-avatar" />
           </div>
        </div>
      </nav>

      {/* View Switcher */}
      {activeView === 'monitors' ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            <StatBox title="Critical Alerts [P1]" value={stats?.critical_alerts || 0} label="Decision Latency: 42ms" icon={ShieldAlert} colorClass="red" />
            <StatBox title="Risk Queue [P2]" value={stats?.suspicious_alerts || 0} label="Top 1k Ranked" icon={Database} colorClass="blue" />
            <StatBox title="Throughput" value={`${Math.round((stats?.throughput_rows_per_sec || 0)/1000)}k`} label="Events / Sec" icon={Zap} colorClass="purple" />
            <StatBox title="Model Precision" value="57.1%" label="Recall @ Top 500" icon={TrendingUp} colorClass="green" />
          </div>
          {renderMonitors()}
        </>
      ) : activeView === 'threats' ? (
        renderThreatIntel()
      ) : activeView === 'intelligence' ? (
        renderIntelligence()
      ) : activeView === 'clusters' ? (
        renderClusters()
      ) : activeView === 'replay' ? (
        renderReplay()
      ) : activeView === 'agent' ? (
        renderAgent()
      ) : activeView === 'reports' ? (
        renderReports()
      ) : (
        renderPlaceholder(activeView)
      )}

    </div>
  );
};

export default SOCDashboard;

/* --- LEAFLET STABLE COMPONENTS --- */

const MapController = ({ center, zoom }) => {
  const map = useMap();
  useEffect(() => {
    if (center && zoom) {
      map.flyTo(center, zoom, {
        animate: true,
        duration: 2.5
      });
    }
  }, [center, zoom, map]);
  return null;
};

const CyberMap = ({ center, zoom }) => {
  const threats = [
    { id: 1, pos: [55.75, 37.61], city: 'Moscow', risk: 'Critical' },
    { id: 2, pos: [52.52, 13.40], city: 'Berlin', risk: 'High' },
    { id: 3, pos: [39.90, 116.40], city: 'Beijing', risk: 'Critical' },
    { id: 4, pos: [52.36, 4.90], city: 'Amsterdam', risk: 'Suspicious' }
  ];

  return (
    <div className="w-full h-full min-h-[500px] relative leaflet-cyber-container">
      <MapContainer 
        center={center} 
        zoom={zoom} 
        scrollWheelZoom={true} 
        className="w-full h-full"
        zoomControl={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
        />
        <MapController center={center} zoom={zoom} />
        
        {threats.map(t => (
          <CircleMarker 
            key={t.id}
            center={t.pos}
            radius={15}
            pathOptions={{ 
              color: t.risk === 'Critical' ? '#ef4444' : '#10b981',
              fillColor: t.risk === 'Critical' ? '#ef4444' : '#10b981',
              fillOpacity: 0.4,
              weight: 2
            }}
          >
            <Popup className="cyber-popup">
              <div className="bg-[#0a0f0d] text-emerald-400 p-2 font-mono text-[10px] uppercase font-black">
                {t.city} NODE: {t.risk}
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
      
      <style dangerouslySetInnerHTML={{ __html: `
        .leaflet-container { 
          background: #090909 !important; 
          border-radius: 24px;
          will-change: filter;
        }
        .leaflet-tile-pane {
           filter: invert(100%) hue-rotate(180deg) brightness(95%) contrast(90%);
           transform: translate3d(0,0,0);
        }
        .leaflet-tile {
          opacity: 1 !important;
        }
        .leaflet-popup-content-wrapper, .leaflet-popup-tip {
          background: #111814 !important;
          color: #10b981 !important;
          border: 1px solid #10b98140;
        }
      `}} />
    </div>
  );
};

