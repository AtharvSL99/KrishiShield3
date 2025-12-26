import { openDB } from 'idb';

const DB_NAME = 'KrishiShieldDB';
const DB_VERSION = 2; // Incrementing version to add new stores

export const initDB = async () => {
  return openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      // 1. Store for Alerts
      if (!db.objectStoreNames.contains('alerts')) {
        db.createObjectStore('alerts', { keyPath: 'id' });
      }
      // 2. Store for General App State
      if (!db.objectStoreNames.contains('appState')) {
        db.createObjectStore('appState', { keyPath: 'key' });
      }
      // 3. NEW: Store for Market Lists (Key: 'Onion', 'Wheat', etc.)
      if (!db.objectStoreNames.contains('marketLists')) {
        db.createObjectStore('marketLists', { keyPath: 'crop' });
      }
      // 4. NEW: Store for Analysis Results (Key: 'Onion_pune', 'Wheat_mumbai')
      if (!db.objectStoreNames.contains('analysisCache')) {
        db.createObjectStore('analysisCache', { keyPath: 'id' });
      }
    },
  });
};

// --- ALERT OPERATIONS ---
export const getStoredAlerts = async () => {
  const db = await initDB();
  const alerts = await db.getAll('alerts');
  return alerts.sort((a, b) => b.id - a.id);
};

export const saveAlertToDB = async (alert) => {
  const db = await initDB();
  return db.put('alerts', alert);
};

// --- MARKET LIST CACHING (Fixes empty dropdowns) ---
export const saveMarketList = async (crop, markets) => {
  const db = await initDB();
  return db.put('marketLists', { crop, markets });
};

export const getMarketList = async (crop) => {
  const db = await initDB();
  const result = await db.get('marketLists', crop);
  return result ? result.markets : [];
};

// --- ANALYSIS RESULT CACHING (Fixes "only last result showing") ---
export const saveAnalysisResult = async (crop, market, data) => {
  const db = await initDB();
  const id = `${crop}_${market}`; // Unique ID like 'Onion_pune'
  return db.put('analysisCache', { id, crop, market, data, timestamp: Date.now() });
};

export const getAnalysisResult = async (crop, market) => {
  const db = await initDB();
  const id = `${crop}_${market}`;
  const result = await db.get('analysisCache', id);
  return result ? result.data : null;
};

// --- GENERIC STATE ---
export const saveAppState = async (key, value) => {
  const db = await initDB();
  return db.put('appState', { key, value });
};

export const getAppState = async (key) => {
  const db = await initDB();
  const result = await db.get('appState', key);
  return result ? result.value : null;
};