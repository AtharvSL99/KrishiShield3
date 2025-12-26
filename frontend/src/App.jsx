import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Sprout, TrendingUp, AlertTriangle, MapPin, Menu, X, Bell, Wifi, WifiOff, MessageSquare, Database,
  Sun, Cloud, CloudRain, CloudLightning, Snowflake, Droplets, Wind, ArrowDown
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { getStoredAlerts, saveAlertToDB, saveAppState, getAppState, saveMarketList, getMarketList, saveAnalysisResult, getAnalysisResult } from './db'; 
import './App.css'; 

function App() {
  // --- STATE ---
  const [crop, setCrop] = useState('Onion');
  const [market, setMarket] = useState('');
  const [marketList, setMarketList] = useState([]);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // UI State
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  // Alert System State
  const [alerts, setAlerts] = useState([]);
  const [lastAlertId, setLastAlertId] = useState(0); 

  // --- LIFECYCLE: Online Status & Init ---
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    if ("Notification" in window) Notification.requestPermission();
    
    // Initial Load
    const loadInitData = async () => {
      // 1. Alerts
      const savedAlerts = await getStoredAlerts();
      setAlerts(savedAlerts);
      if(savedAlerts.length > 0) setLastAlertId(savedAlerts[0].id);

      // 2. Restore last selection
      const savedCrop = await getAppState('lastCrop');
      const savedMarket = await getAppState('lastMarket');
      if(savedCrop) setCrop(savedCrop);
      if(savedMarket) setMarket(savedMarket);

      // 3. If offline, immediately try to load market list for default crop
      if (!navigator.onLine) {
        const offlineMarkets = await getMarketList(savedCrop || 'Onion');
        setMarketList(offlineMarkets);
      }
    };

    loadInitData();

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // --- SCHEDULER: Poll for Alerts ---
  useEffect(() => {
    const pollAlerts = async () => {
      if (!isOnline) return;
      try {
        const res = await axios.get('http://127.0.0.1:8000/alerts');
        const latestAlerts = res.data;
        if (latestAlerts.length > 0) {
          const newest = latestAlerts[0];
          if (newest.id > lastAlertId) {
             if (Notification.permission === "granted") {
               new Notification(`Risk Alert: ${newest.crop}`, { body: newest.message, icon: '/vite.svg' });
             }
             setAlerts(latestAlerts);
             setLastAlertId(newest.id);
             for (const alert of latestAlerts) await saveAlertToDB(alert); 
          }
        }
      } catch (e) { console.error("Background sync failed", e); }
    };
    pollAlerts();
    const interval = setInterval(pollAlerts, 30000);
    return () => clearInterval(interval);
  }, [isOnline, lastAlertId]);

  // --- MARKET FETCHING (Online + Offline Fallback) ---
  useEffect(() => {
    const fetchMarkets = async () => {
      try {
        setMarketList([]); 
        
        if (isOnline) {
          // ONLINE: Fetch from API
          const response = await axios.get(`http://127.0.0.1:8000/markets/${crop}`);
          const markets = response.data.markets;
          setMarketList(markets);
          
          // Cache for offline use
          await saveMarketList(crop, markets);
          
          // Default selection if none selected
          if (!market && markets.length > 0) setMarket(markets[0]);
        } else {
          // OFFLINE: Load from IDB
          const cachedMarkets = await getMarketList(crop);
          if (cachedMarkets.length > 0) {
            setMarketList(cachedMarkets);
          } else {
            setError(`No offline data for ${crop}. Connect to internet.`);
          }
        }
      } catch (err) { setError("Could not load markets."); }
    };
    fetchMarkets();
  }, [crop, isOnline]);

  // --- ANALYZE ACTION (Online + Offline Fallback) ---
  const handleAnalyze = async () => {
    if (!market) return alert("Select market");
    
    setLoading(true); setError(''); setData(null);
    if (window.innerWidth <= 1024) setIsSidebarOpen(false);
    
    // Save selection pref
    await saveAppState('lastCrop', crop);
    await saveAppState('lastMarket', market);

    if (isOnline) {
      // --- ONLINE MODE ---
      try {
        const response = await axios.post('http://127.0.0.1:8000/analyze', { crop, market });
        const result = response.data;
        setData(result);
        // Cache result for this specific combo
        await saveAnalysisResult(crop, market, result);
      } catch (err) { 
          setError(err.response?.data?.detail || "Prediction failed."); 
      } finally { 
          setLoading(false); 
      }
    } else {
      // --- OFFLINE MODE ---
      try {
        // Try to find cached result for this specific combo
        const cachedResult = await getAnalysisResult(crop, market);
        if (cachedResult) {
          setData(cachedResult);
        } else {
          setError("This specific market hasn't been analyzed/cached yet.");
        }
      } catch (e) {
        setError("Error loading offline data.");
      } finally {
        setLoading(false);
      }
    }
  };

  const getRiskColor = (level) => level.includes("HIGH") ? '#dc2626' : level.includes("MODERATE") ? '#d97706' : '#16a34a';

  const getDateLabel = (dateString) => {
    const d = new Date(dateString); const today = new Date(); d.setHours(0,0,0,0); today.setHours(0,0,0,0);
    const diffDays = Math.ceil((d - today) / (86400000));
    if (diffDays === -1) return "Yesterday"; if (diffDays === 0) return "Today"; if (diffDays === 1) return "Tomorrow";
    return d.toLocaleDateString(undefined, {weekday: 'short'});
  };

  // --- Weather Helpers ---
  const getWeatherIcon = (code) => {
    if (code >= 95) return <CloudLightning size={36} color="#6366f1" />;
    if (code >= 71) return <Snowflake size={36} color="#3b82f6" />;
    if (code >= 51) return <CloudRain size={36} color="#0ea5e9" />;
    if (code >= 1) return <Cloud size={36} color="#94a3b8" />;
    return <Sun size={36} color="#f59e0b" />;
  };

  const getWeatherLabel = (code) => {
    if (code >= 95) return "Stormy";
    if (code >= 71) return "Snow";
    if (code >= 51) return "Rainy";
    if (code >= 1) return "Cloudy";
    return "Sunny";
  };

  return (
    <div className="app-container">
      {/* Mobile Menus */}
      <button className="menu-btn" onClick={() => setIsSidebarOpen(true)}><Menu size={24} /></button>
      <div className={`overlay ${isSidebarOpen ? 'active' : ''}`} onClick={() => setIsSidebarOpen(false)}></div>

      {/* Notifications */}
      {showNotifications && (
        <div style={{ position: 'fixed', top: '70px', right: '20px', width: '350px', maxHeight: '500px', background: 'white', border: '1px solid #e2e8f0', borderRadius: '12px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', zIndex: 60, overflowY: 'auto', padding: '1rem' }}>
          <div style={{display:'flex', justifyContent:'space-between', marginBottom:'1rem'}}>
            <h3 style={{margin:0, color:'#1e293b'}}>Alert History</h3>
            <button onClick={() => setShowNotifications(false)} style={{background:'none', border:'none', cursor:'pointer'}}><X size={18}/></button>
          </div>
          {alerts.length === 0 ? <p style={{color:'#94a3b8'}}>No recent alerts.</p> : (
            alerts.map((alert) => (
              <div key={alert.id} style={{ marginBottom:'10px', padding:'10px', borderRadius:'8px', background: alert.type === 'Critical' ? '#fef2f2' : '#fffbeb', borderLeft: `4px solid ${alert.type === 'Critical' ? '#ef4444' : '#f59e0b'}` }}>
                <div style={{display:'flex', justifyContent:'space-between', fontSize:'0.8rem', color:'#64748b'}}>
                  <span>{alert.crop} • {alert.market}</span>
                  <span>{alert.timestamp.split(' ')[1]}</span>
                </div>
                <p style={{margin:'5px 0', fontSize:'0.9rem', color:'#334155'}}>{alert.message}</p>
                {alert.is_sms_sent && <div style={{display:'flex', gap:'5px', alignItems:'center', fontSize:'0.75rem', color:'#059669', marginTop:'5px'}}><MessageSquare size={12}/> SMS Sent</div>}
              </div>
            ))
          )}
        </div>
      )}

      {/* Sidebar */}
      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h2><img src="/Krishi.png" alt="Krishi Shield Logo" style={{width: '24px', height: '24px', marginRight: '8px', verticalAlign: 'middle'}} /> Krishi Shield</h2>
          <button className="close-btn" onClick={() => setIsSidebarOpen(false)}><X size={24} /></button>
        </div>
        <div className="form-group">
          <label>Select Crop</label>
          <select value={crop} onChange={(e) => setCrop(e.target.value)} className="styled-select">
            <option value="Onion">Onion</option><option value="Wheat">Wheat</option><option value="Potato">Potato</option>
          </select>
        </div>
        <div className="form-group">
          <label>Select Market</label>
          <select value={market} onChange={(e) => setMarket(e.target.value)} className="styled-select" disabled={marketList.length === 0}>
            {marketList.length === 0 ? <option>Loading...</option> : marketList.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <button className="analyze-btn" onClick={handleAnalyze} disabled={loading || !market}>
          {loading ? 'Analyzing...' : isOnline ? 'Analyze Future Risk' : 'Load Offline Data'}
        </button>
        {!isOnline && (
          <div style={{marginTop:'15px', padding:'10px', background:'rgba(255,255,255,0.1)', borderRadius:'8px', border:'1px solid rgba(255,255,255,0.2)'}}>
            <p style={{margin:0, fontSize:'0.85rem', color:'#fca5a5', fontWeight:'bold', display:'flex', alignItems:'center', gap:'5px'}}>
               <Database size={14}/> Offline Mode
            </p>
            <p style={{margin:'5px 0 0 0', fontSize:'0.75rem', color:'#e2e8f0'}}>
              You can only view markets you have previously analyzed.
            </p>
          </div>
        )}
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <div className="header" style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
          <div>
            <h1><Sprout color="#15803d" size={40} /> {crop} Future Risk Calculator</h1>
            <p className="subtitle">AI-Driven Market Analysis • {market ? market : 'Select Market'}</p>
          </div>
          <div style={{display:'flex', gap:'15px', alignItems:'center'}}>
            <div style={{display:'flex', alignItems:'center', gap:'5px', color: isOnline ? '#16a34a' : '#ef4444', fontWeight:'bold', fontSize:'0.9rem'}}>
               {isOnline ? <Wifi size={18} /> : <WifiOff size={18} />}
               {isOnline ? "Online" : "Offline"}
            </div>
            <button onClick={() => setShowNotifications(!showNotifications)} style={{position:'relative', background:'white', border:'1px solid #e2e8f0', padding:'8px', borderRadius:'50%', cursor:'pointer'}}>
              <Bell size={20} color="#334155" />
              {alerts.length > 0 && <span style={{position:'absolute', top:'-2px', right:'-2px', background:'#ef4444', color:'white', fontSize:'10px', width:'16px', height:'16px', borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center'}}>{alerts.length}</span>}
            </button>
          </div>
        </div>
        <hr style={{ borderColor: '#e2e8f0', margin: '2rem 0' }} />

        {data ? (
          <div className="fade-in"> 
            <div className="metrics-grid">
              
              {/* Card 1: Projected Price */}
              <div className="metric-card">
                <p style={{ color: '#64748b' }}>Projected Price</p>
                <div className="price-value">₹ {data.projected_price.toLocaleString()}</div>
                <div style={{ marginTop: '10px', fontWeight: '600', color: data.price_change > 0 ? '#dc2626' : '#16a34a', display: 'flex', alignItems: 'center', gap: '5px'}}>
                  <TrendingUp size={18} /> {data.price_change}% vs Norm
                </div>
              </div>

              {/* Card 2: Risk Level */}
              <div className="metric-card" style={{ textAlign: 'center' }}>
                <p style={{ color: '#64748b' }}>Risk Level</p>
                <div className="risk-value" style={{ color: getRiskColor(data.risk_level) }}>
                  {data.risk_level} <span style={{ fontSize: '0.6em', opacity: 0.8 }}>({data.risk_score}%)</span>
                </div>
                <div className="risk-bar">
                  <div className="risk-fill" style={{ width: `${data.risk_score}%`, backgroundColor: getRiskColor(data.risk_level), height: '100%', borderRadius: '6px', transition: 'width 1s ease'}}></div>
                </div>
              </div>

              {/* Card 3: EXPANDED Weather Card */}
              <div className="metric-card weather-card">
                 <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
                    <div>
                      <p style={{ color: '#64748b', marginBottom:'5px' }}>Current Weather</p>
                      <div style={{ fontSize: '2.5rem', fontWeight: '800', color: '#1e293b' }}>
                        {data.timeline[0]?.temperature_2m_max}°C
                      </div>
                      <p style={{ color: '#334155', fontWeight:'600', display:'flex', alignItems:'center', gap:'5px', marginTop:'5px' }}>
                         {getWeatherLabel(data.timeline[0]?.weather_code || 0)}
                      </p>
                    </div>
                    <div style={{background:'#f0f9ff', padding:'10px', borderRadius:'12px'}}>
                      {getWeatherIcon(data.timeline[0]?.weather_code || 0)}
                    </div>
                 </div>
                 
                 {/* New: Weather Grid for Params */}
                 <div style={{marginTop:'15px', paddingTop:'15px', borderTop:'1px solid #f1f5f9', display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:'5px'}}>
                    
                    <div style={{textAlign:'center'}}>
                       <Droplets size={16} color="#0ea5e9" style={{marginBottom:'4px'}}/> 
                       <div style={{fontSize:'0.85rem', fontWeight:'bold', color:'#334155'}}>{data.timeline[0]?.precipitation_sum || 0}mm</div>
                       <div style={{fontSize:'0.7rem', color:'#64748b'}}>Rain</div>
                    </div>

                    <div style={{textAlign:'center', borderLeft:'1px solid #f1f5f9', borderRight:'1px solid #f1f5f9'}}>
                       <Wind size={16} color="#64748b" style={{marginBottom:'4px'}}/> 
                       <div style={{fontSize:'0.85rem', fontWeight:'bold', color:'#334155'}}>{data.timeline[0]?.wind_speed_10m_max || 0}</div>
                       <div style={{fontSize:'0.7rem', color:'#64748b'}}>km/h</div>
                    </div>

                    <div style={{textAlign:'center'}}>
                       <ArrowDown size={16} color="#0ea5e9" style={{marginBottom:'4px'}}/> 
                       <div style={{fontSize:'0.85rem', fontWeight:'bold', color:'#334155'}}>{data.timeline[0]?.temperature_2m_min || 0}°C</div>
                       <div style={{fontSize:'0.7rem', color:'#64748b'}}>Min</div>
                    </div>

                 </div>
              </div>

            </div>

            <div style={{ background: 'white', padding: '1.5rem', borderRadius: '16px', border: '1px solid #e2e8f0', marginBottom: '2rem', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
              <h3 style={{ color: '#334155', marginTop: 0, marginBottom: '1.5rem' }}>Market Price Trend (3 Months)</h3>
              <div style={{ width: '100%', height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data.price_trend}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="date" tick={{fontSize: 12}} />
                    <YAxis domain={['auto', 'auto']} tick={{fontSize: 12}} />
                    <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
                    <Line type="monotone" dataKey="price" stroke="#16a34a" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {data.advisory && (
                <div className={`advisory-box ${data.advisory.color}`} style={{ padding: '1.5rem', borderRadius: '12px', marginBottom: '2rem', borderLeft: '5px solid', background: 'white', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderColor: data.advisory.color === 'success' ? '#22c55e' : data.advisory.color === 'warning' ? '#eab308' : '#ef4444' }}>
                    <h3 style={{ marginTop: 0, color: '#334155' }}>{data.advisory.title}</h3>
                    <p>{data.advisory.body}</p>
                    <ul style={{ paddingLeft: '1.5rem', marginBottom: 0 }}>
                        {data.advisory.steps.map((step, idx) => (<li key={idx}>{step}</li>))}
                    </ul>
                </div>
            )}

            <h3 style={{ color: '#334155', marginTop:'2rem' }}>5-Day Weather Forecast & Impact</h3>
            <div className="data-table-container">
              <table>
                <thead><tr><th>Day</th><th>Date</th><th>Max Temp (°C)</th><th>Precipitation (mm)</th></tr></thead>
                <tbody>
                  {data.timeline.map((row, idx) => {
                    const label = getDateLabel(row.Date); const isToday = label === "Today";
                    return (
                      <tr key={idx} style={isToday ? { backgroundColor: '#dcfce7' } : {}}>
                        <td style={{ fontWeight: isToday ? '800' : 'normal', color: isToday ? '#166534' : '#334155' }}>{label}</td>
                        <td style={{ color: '#64748b', fontSize: '0.9rem' }}>{new Date(row.Date).toLocaleDateString(undefined, {month:'short', day:'numeric'})}</td>
                        <td style={{ fontWeight: 'bold', fontSize: isToday ? '1.1rem' : '1rem' }}>{row.temperature_2m_max}</td>
                        <td>{row.precipitation_sum > 0 ? (<span style={{color:'#2563eb', fontWeight:'bold'}}>{row.precipitation_sum}</span>) : <span style={{color:'#94a3b8'}}>-</span>}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', marginTop: '5rem', color: '#94a3b8' }}>
            <MapPin size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
            <p style={{ fontSize: '1.2rem' }}>{isOnline ? "Select a crop and market to begin analysis." : "You are offline. Showing last cached data if available."}</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;