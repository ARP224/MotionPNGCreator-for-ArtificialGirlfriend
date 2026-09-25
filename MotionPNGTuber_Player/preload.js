/**
 * MotionPNGTuber Player - Electron Preload Script
 *
 * Exposes a safe API to the renderer process for:
 * - IPC communication with main process
 * - File system access (through main process)
 * - Window controls
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // Receive initialization data from main process
    onInit: (callback) => {
        ipcRenderer.on('init', (event, data) => callback(data));
    },

    // Resize window (scroll wheel)
    resizeWindow: (delta) => {
        ipcRenderer.send('resize-window', delta);
    },

    // Set scale directly (from slider)
    setScale: (scale) => {
        ipcRenderer.send('set-scale', scale);
    },

    // Toggle resize slider visibility
    onToggleResizeSlider: (callback) => {
        ipcRenderer.on('toggle-resize-slider', (event, scale) => callback(scale));
    },

    // Receive scale changed notification
    onScaleChanged: (callback) => {
        ipcRenderer.on('scale-changed', (event, scale) => callback(scale));
    },

    // Set base window size (when video dimensions are known)
    setBaseSize: (width, height) => {
        ipcRenderer.send('set-base-size', width, height);
    },

    // Load character folder and get motion data
    loadCharacterFolder: (folderPath) => {
        return ipcRenderer.invoke('load-character-folder', folderPath);
    },

    // Load single asset folder (flat structure)
    loadSingleAsset: (folderPath) => {
        return ipcRenderer.invoke('load-single-asset', folderPath);
    },

    // Read file content
    readFile: (filePath) => {
        return ipcRenderer.invoke('read-file', filePath);
    },

    // Window drag (move window)
    startDrag: () => {
        ipcRenderer.send('window-drag-start');
    },
    dragWindow: (deltaX, deltaY) => {
        ipcRenderer.send('window-drag-move', deltaX, deltaY);
    },

    // Test mouth animation (external control)
    onTestMouth: (callback) => {
        ipcRenderer.on('test-mouth', (event, enabled) => callback(enabled));
    },

    // Load playlist JSON for review mode
    loadPlaylist: (jsonPath) => {
        return ipcRenderer.invoke('load-playlist', jsonPath);
    }
});

console.log('[Preload] electronAPI exposed to renderer');
