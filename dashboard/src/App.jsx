import React, { useState, useEffect } from 'react';
import { 
  ShieldAlert, Shield, ShieldQuestion, 
  Search, Filter, ShieldOff, BrainCircuit, Activity
} from 'lucide-react';
import AlertTable from './components/AlertTable';

function App() {
  const [alerts, setAlerts] = useState([]);
  const [stats, setStats] = useState({ critical: 0, high: 0, total: 0 });
  const [activeQueue, setActiveQueue] = useState('all'); // all, primary, secondary, anomaly
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    // Load the pipeline output from JSON
    fetch('/soc_alerts.json')
      .then(res => res.json())
      .then(data => {
        setAlerts(data);
        const critical = data.filter(d => d.alert_severity === 'CRITICAL').length;
        const high = data.filter(d => d.alert_severity === 'HIGH').length;
        setStats({ critical, high, total: data.length });
      })
      .catch(err => console.error("Error loading alerts:", err));
  }, []);

  const getFilteredAlerts = () => {
    let filtered = alerts;
    
    // Search
    if (searchTerm) {
      const lowerTheme = searchTerm.toLowerCase();
      filtered = filtered.filter(a => String(a['User ID']).includes(lowerTheme) || String(a['IP Address']).includes(lowerTheme));
    }
    
    // Queue Filtering
    if (activeQueue === 'primary') {
      filtered = filtered.sort((a, b) => b.model_score - a.model_score);
    } else if (activeQueue === 'secondary') {
      filtered = filtered.filter(a => a.high_model_high_ae).sort((a, b) => (b.model_score + b.ae_score) - (a.model_score + a.ae_score));
    } else if (activeQueue === 'anomaly') {
      filtered = filtered.sort((a, b) => b.ae_score - a.ae_score);
    }
    
    return filtered;
  };

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <header className="top-bar glass-header">
        <div className="brand">
          <ShieldAlert size={28} color="#3b82f6" />
          ChronoBase SOC
        </div>
        <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
          <div style={{ position: 'relative' }}>
            <Search size={18} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input 
              type="text" 
              placeholder="Search User ID or IP..." 
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              style={{
                background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-color)', 
                color: 'var(--text-primary)', padding: '0.5rem 1rem 0.5rem 2.5rem', 
                borderRadius: '999px', width: '250px', outline: 'none'
              }}
            />
          </div>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <div className="badge critical">
              <span className="live-indicator"></span>
              {stats.critical} Critical
            </div>
            <div className="badge high">{stats.high} High</div>
          </div>
        </div>
      </header>
      
      <main className="main-content">
        {/* Left Sidebar / Queues */}
        <aside className="sidebar">
          <h3 style={{ textTransform: 'uppercase', fontSize: '0.8rem', color: 'var(--text-muted)', letterSpacing: '0.05em', marginBottom: '0.5rem', paddingLeft: '0.5rem' }}>
            Alert Queues
          </h3>
          
          <div className={`queue-card glass-panel ${activeQueue === 'all' ? 'active' : ''}`} onClick={() => setActiveQueue('all')}>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <div className="queue-icon" style={{ background: 'rgba(107, 114, 128, 0.2)' }}>
                <Activity size={20} color="#9ca3af" />
              </div>
              <div>
                <h4 style={{ fontSize: '0.95rem' }}>Global Feed</h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>All scored events</p>
              </div>
            </div>
          </div>

          <div className={`queue-card glass-panel ${activeQueue === 'primary' ? 'active' : ''}`} onClick={() => setActiveQueue('primary')}>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <div className="queue-icon" style={{ background: 'rgba(239, 68, 68, 0.15)' }}>
                <ShieldOff size={20} color="#ef4444" />
              </div>
              <div>
                <h4 style={{ fontSize: '0.95rem' }}>Primary Queue</h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Highest Model Score (ATO)</p>
              </div>
            </div>
          </div>

          <div className={`queue-card glass-panel ${activeQueue === 'secondary' ? 'active' : ''}`} onClick={() => setActiveQueue('secondary')}>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <div className="queue-icon" style={{ background: 'rgba(245, 158, 11, 0.15)' }}>
                <ShieldQuestion size={20} color="#f59e0b" />
              </div>
              <div>
                <h4 style={{ fontSize: '0.95rem' }}>Secondary Queue</h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>High Model + High Anomaly</p>
              </div>
            </div>
          </div>

          <div className={`queue-card glass-panel ${activeQueue === 'anomaly' ? 'active' : ''}`} onClick={() => setActiveQueue('anomaly')}>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <div className="queue-icon" style={{ background: 'rgba(59, 130, 246, 0.15)' }}>
                <BrainCircuit size={20} color="#3b82f6" />
              </div>
              <div>
                <h4 style={{ fontSize: '0.95rem' }}>Anomaly Queue</h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Pure behavioral drift</p>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Table Area */}
        <section className="table-container glass-panel">
          <div className="table-header">
            <h2 style={{ fontSize: '1.2rem', fontWeight: 600 }}>
              {activeQueue === 'all' && 'Global Alert Feed'}
              {activeQueue === 'primary' && 'Likely Account Takeovers'}
              {activeQueue === 'secondary' && 'Complex Threats (ATO + Drift)'}
              {activeQueue === 'anomaly' && 'Behavioral Anomalies'}
            </h2>
            <div className="badge" style={{ background: 'var(--bg-tertiary)' }}>
              Showing {getFilteredAlerts().length} results
            </div>
          </div>
          <div className="table-scroll">
            <AlertTable alerts={getFilteredAlerts()} />
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
