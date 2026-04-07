import React from 'react';

const ScoreBar = ({ score, color }) => {
  return (
    <div className="score-bar-container">
      <div 
        className="score-bar" 
        style={{ width: `${Math.max(score * 100, 5)}%`, backgroundColor: color }}
      ></div>
    </div>
  );
};

const AlertTable = ({ alerts }) => {
  if (!alerts || alerts.length === 0) {
    return (
      <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        No alerts match criteria.
      </div>
    );
  }

  const formatTimestamp = (ts) => {
    if (!ts) return '-';
    // If it's a string, just display it
    if (typeof ts === 'string') return ts.split('.')[0];
    const d = new Date(ts * 1000);
    return d.toLocaleString();
  };

  return (
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>User Info</th>
          <th>Context</th>
          <th>Scores (M / A / R)</th>
          <th>Risk</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody>
        {alerts.map((a, i) => (
          <tr key={i}>
            <td style={{ whiteSpace: 'nowrap', color: 'var(--text-secondary)' }}>
              {formatTimestamp(a['Login Timestamp'])}
            </td>
            <td>
              <div style={{ fontWeight: 500, color: 'var(--text-primary)' }}>User {a['User ID']}</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{a['IP Address']}</div>
            </td>
            <td>
              <div style={{ fontSize: '0.85rem' }}>{a.Country} • {a['Device Type']}</div>
              <div style={{ fontSize: '0.8rem', color: a['Login Successful'] ? 'var(--success)' : 'var(--danger)' }}>
                {a['Login Successful'] ? 'Success' : 'Failed'}
              </div>
            </td>
            <td>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem' }}>
                  <span style={{ width: '12px', color: 'var(--danger)' }}>M</span> 
                  <ScoreBar score={a.model_score} color="var(--danger)" />
                  <span style={{ minWidth: '25px', textAlign: 'right' }}>{(a.model_score * 100).toFixed(0)}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem' }}>
                  <span style={{ width: '12px', color: 'var(--info)' }}>A</span> 
                  <ScoreBar score={a.ae_score} color="var(--info)" />
                  <span style={{ minWidth: '25px', textAlign: 'right' }}>{(a.ae_score * 100).toFixed(0)}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem' }}>
                  <span style={{ width: '12px', color: 'var(--warning)' }}>R</span> 
                  <ScoreBar score={a.rule_score} color="var(--warning)" />
                  <span style={{ minWidth: '25px', textAlign: 'right' }}>{(a.rule_score * 100).toFixed(0)}</span>
                </div>
              </div>
            </td>
            <td>
              <div className={`badge ${a.alert_severity?.toLowerCase() || 'low'}`}>
                {a.alert_severity || 'LOW'} 
                <span style={{ marginLeft: '4px', opacity: 0.8 }}>
                  ({a.final_risk ? a.final_risk.toFixed(2) : '0.00'})
                </span>
              </div>
            </td>
            <td style={{ fontSize: '0.8rem', maxWidth: '250px', color: 'var(--text-secondary)' }}>
              {a.reason_codes ? a.reason_codes : 
                (a.model_score > 0.8 ? "Predicted ATO" : "Normal")}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default AlertTable;
