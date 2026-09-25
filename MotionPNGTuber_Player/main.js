/**
 * MotionPNGTuber Player - Electron Main Process
 *
 * Transparent window application for displaying character animation
 * with lip-sync synchronized to AG's TTS audio.
 */

const { app, BrowserWindow, ipcMain, Menu, protocol, net } = require('electron');
const path = require('path');
const fs = require('fs');
const { pathToFileURL } = require('url');

// Register local:// as a privileged scheme (must be done before app ready)
// so that media/fetch works with webSecurity enabled
protocol.registerSchemesAsPrivileged([
    {
        scheme: 'local',
        privileges: {
            standard: true,
            secure: true,
            supportFetchAPI: true,
            stream: true,
            bypassCSP: true
        }
    }
]);

// Increase memory limits for handling large audio data
app.commandLine.appendSwitch('js-flags', '--max-old-space-size=512');

// Disable GPU sandbox to prevent renderer crashes with audio processing
app.commandLine.appendSwitch('disable-gpu-sandbox');

// Enable hardware acceleration for video
app.commandLine.appendSwitch('enable-gpu-rasterization');

let win = null;
let scale = 1.0;

// Base window size (will be adjusted based on video dimensions)
let BASE_WIDTH = 540;
let BASE_HEIGHT = 960;

// Settings file path (in app directory)
const settingsPath = path.join(__dirname, 'settings.json');

// Load settings from file
function loadSettings() {
    try {
        if (fs.existsSync(settingsPath)) {
            const data = fs.readFileSync(settingsPath, 'utf8');
            const settings = JSON.parse(data);
            if (typeof settings.scale === 'number') {
                scale = Math.max(0.3, Math.min(2.0, settings.scale));
                console.log('[Main] Loaded scale from settings:', scale);
            }
        }
    } catch (error) {
        console.error('[Main] Failed to load settings:', error);
    }
}

// Save settings to file
function saveSettings() {
    try {
        const settings = { scale };
        fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2), 'utf8');
        console.log('[Main] Saved settings, scale:', scale);
    } catch (error) {
        console.error('[Main] Failed to save settings:', error);
    }
}

// Command line arguments
let wsPort = 8765;
let characterFolder = '';
let testMouthOnStart = false;
let playlistPath = '';

// Parse command line arguments
function parseArgs() {
    const args = process.argv.slice(2);
    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--ws-port' && args[i + 1]) {
            wsPort = parseInt(args[i + 1], 10);
            i++;
        }
        if (args[i] === '--character-folder' && args[i + 1]) {
            characterFolder = args[i + 1];
            i++;
        }
        if (args[i] === '--test-mouth') {
            testMouthOnStart = true;
        }
        if (args[i] === '--playlist' && args[i + 1]) {
            playlistPath = args[i + 1];
            i++;
        }
    }
    console.log('[Main] WebSocket port:', wsPort);
    console.log('[Main] Character folder:', characterFolder);
    console.log('[Main] Playlist:', playlistPath);
    console.log('[Main] Test mouth:', testMouthOnStart);
}

// Register custom protocol for local file access
function registerLocalProtocol() {
    protocol.handle('local', (request) => {
        const url = new URL(request.url);
        // standardスキームでは local://C:/... のドライブレターがhostとして
        // 解釈される（local://c/...）ため、host + ':' + pathname で復元する
        // パスに生の % が含まれると decodeURIComponent が例外になるため、
        // デコード失敗時はそのまま使う（rendererはエンコードせず生成している）
        let filePath;
        try {
            filePath = decodeURIComponent(url.pathname);
        } catch (e) {
            filePath = url.pathname;
        }
        if (url.host) {
            filePath = `${url.host}:${filePath}`;
        } else if (/^\/[A-Za-z]:\//.test(filePath)) {
            // local:///C:/... 形式（hostなし）は先頭のスラッシュを除去
            filePath = filePath.slice(1);
        }
        return net.fetch(pathToFileURL(filePath).toString());
    });
}

// Create the transparent window
function createWindow() {
    // Apply saved scale to initial size
    const initialWidth = Math.max(100, Math.round(BASE_WIDTH * scale));
    const initialHeight = Math.max(100, Math.round(BASE_HEIGHT * scale));

    win = new BrowserWindow({
        width: initialWidth,
        height: initialHeight,
        minWidth: 50,
        minHeight: 50,
        frame: false,
        transparent: true,
        alwaysOnTop: true,
        resizable: false,
        hasShadow: false,
        skipTaskbar: false,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            webSecurity: true  // local files are served via privileged local:// scheme
        }
    });

    // Load the HTML file
    win.loadFile('index-electron.html');

    // Send initialization data when page is ready
    win.webContents.on('did-finish-load', () => {
        win.webContents.send('init', {
            wsPort: wsPort,
            characterFolder: characterFolder,
            playlistPath: playlistPath
        });
        // Auto-start test mouth if launched with --test-mouth
        if (testMouthOnStart) {
            setTimeout(() => {
                win.webContents.send('test-mouth', true);
            }, 2000);  // Wait for assets to load
        }
    });

    // Handle right-click context menu directly in main process
    win.webContents.on('context-menu', (event, params) => {
        event.preventDefault();
        contextMenu.popup({ window: win });
    });

    // Open DevTools in development mode (F12 key)
    win.webContents.on('before-input-event', (event, input) => {
        if (input.key === 'F12') {
            win.webContents.toggleDevTools();
        }
    });

    // Handle window close
    win.on('closed', () => {
        win = null;
    });
}

// Context menu
const contextMenu = Menu.buildFromTemplate([
    {
        label: 'Minimize',
        click: () => {
            if (win) win.minimize();
        }
    },
    { type: 'separator' },
    {
        label: 'Always on Top',
        type: 'checkbox',
        checked: true,
        click: (menuItem) => {
            if (win) win.setAlwaysOnTop(menuItem.checked);
        }
    },
    { type: 'separator' },
    {
        label: 'Resize',
        click: () => {
            if (win) {
                win.webContents.send('toggle-resize-slider', scale);
            }
        }
    },
    {
        label: 'Reset Size',
        click: () => {
            scale = 1.0;
            if (win) {
                const [x, y] = win.getPosition();
                win.setBounds({ x, y, width: BASE_WIDTH, height: BASE_HEIGHT });
                win.webContents.send('scale-changed', scale);
            }
            saveSettings();
        }
    },
    { type: 'separator' },
    {
        label: 'Quit',
        click: () => {
            app.quit();
        }
    }
]);

// IPC Handlers

// Apply current scale to window (fix bottom-left corner)
function applyScaleToWindow() {
    if (!win) return;
    try {
        const [currentX, currentY] = win.getPosition();
        const [, currentHeight] = win.getSize();
        const newWidth = Math.max(100, Math.round(BASE_WIDTH * scale));
        const newHeight = Math.max(100, Math.round(BASE_HEIGHT * scale));

        // Fix bottom-left corner: calculate new Y so bottom stays in place
        const bottomY = currentY + currentHeight;
        const newY = bottomY - newHeight;

        win.setBounds({ x: currentX, y: newY, width: newWidth, height: newHeight });
    } catch (error) {
        console.error('[Main] Error setting scale:', error);
    }
}

// Debounced settings save (連続リサイズ中のファイル書き込みを抑制)
let saveSettingsTimer = null;
function saveSettingsDebounced() {
    if (saveSettingsTimer) clearTimeout(saveSettingsTimer);
    saveSettingsTimer = setTimeout(() => {
        saveSettingsTimer = null;
        saveSettings();
    }, 500);
}

// Resize window (scroll to resize)
ipcMain.on('resize-window', (event, delta) => {
    scale = Math.max(0.3, Math.min(2.0, scale + delta));
    applyScaleToWindow();
    saveSettingsDebounced();
});

// Set scale directly (from slider)
ipcMain.on('set-scale', (event, newScale) => {
    scale = Math.max(0.3, Math.min(2.0, newScale));
    applyScaleToWindow();
    saveSettingsDebounced();
});

// Set base window size (called when video dimensions are known)
ipcMain.on('set-base-size', (event, width, height) => {
    console.log('[Main] set-base-size called with:', width, 'x', height);
    BASE_WIDTH = width;
    BASE_HEIGHT = height;
    if (win) {
        const [x, y] = win.getPosition();
        const newWidth = Math.round(BASE_WIDTH * scale);
        const newHeight = Math.round(BASE_HEIGHT * scale);
        console.log('[Main] Applying base size with scale:', scale, '-> window:', newWidth, 'x', newHeight);
        win.setBounds({ x, y, width: newWidth, height: newHeight });
    }
});

// Window drag handlers
let dragStartPos = null;

ipcMain.on('window-drag-start', () => {
    if (win) {
        dragStartPos = win.getPosition();
    }
});

ipcMain.on('window-drag-move', (event, deltaX, deltaY) => {
    if (win && dragStartPos) {
        win.setPosition(dragStartPos[0] + deltaX, dragStartPos[1] + deltaY);
    }
});

// Load character folder and return motion data
ipcMain.handle('load-character-folder', async (event, folderPath) => {
    console.log('[Main] Loading character folder:', folderPath);

    if (!folderPath || !fs.existsSync(folderPath)) {
        console.error('[Main] Character folder not found:', folderPath);
        return { error: 'Folder not found', motions: [] };
    }

    try {
        const motions = [];
        const subfolders = fs.readdirSync(folderPath);

        // Check for common mouth folder at root
        const commonMouthPath = path.join(folderPath, 'mouth');
        const hasCommonMouth = fs.existsSync(commonMouthPath) &&
                              fs.statSync(commonMouthPath).isDirectory();

        console.log('[Main] Common mouth folder:', hasCommonMouth ? commonMouthPath : 'none');

        for (const subfolder of subfolders) {
            if (subfolder === 'mouth') continue;  // Skip mouth folder

            const motionPath = path.join(folderPath, subfolder);

            // Skip if not a directory
            if (!fs.statSync(motionPath).isDirectory()) continue;

            const files = fs.readdirSync(motionPath);

            // Find mouthless video (WebM or MP4)
            const videoFile = files.find(f =>
                f.includes('mouthless') &&
                (f.endsWith('.webm') || f.endsWith('.mp4'))
            );

            // Find mouth_track.json
            const trackFile = files.find(f => f === 'mouth_track.json');

            if (videoFile && trackFile) {
                // Check for motion-specific mouth folder
                const motionMouthPath = path.join(motionPath, 'mouth');
                const hasMotionMouth = fs.existsSync(motionMouthPath) &&
                                       fs.statSync(motionMouthPath).isDirectory();

                const mouthPath = hasMotionMouth ? motionMouthPath :
                                  (hasCommonMouth ? commonMouthPath : null);

                if (!mouthPath) {
                    console.warn('[Main] No mouth folder found for motion:', subfolder);
                    continue;
                }

                // Verify required mouth sprites exist
                const requiredSprites = ['closed.png', 'open.png'];
                const hasRequiredSprites = requiredSprites.every(sprite =>
                    fs.existsSync(path.join(mouthPath, sprite))
                );

                if (!hasRequiredSprites) {
                    console.warn('[Main] Missing required mouth sprites for:', subfolder);
                    continue;
                }

                // Get optional mouth sprites
                const optionalSprites = ['half.png', 'e.png', 'u.png'];
                const availableSprites = ['closed.png', 'open.png'];
                for (const sprite of optionalSprites) {
                    if (fs.existsSync(path.join(mouthPath, sprite))) {
                        availableSprites.push(sprite);
                    }
                }

                motions.push({
                    name: subfolder,
                    videoPath: path.join(motionPath, videoFile),
                    trackPath: path.join(motionPath, trackFile),
                    mouthPath: mouthPath,
                    availableSprites: availableSprites
                });

                console.log('[Main] Found motion:', subfolder, '- Video:', videoFile);
            }
        }

        console.log('[Main] Total motions found:', motions.length);
        return { motions: motions };

    } catch (error) {
        console.error('[Main] Error loading character folder:', error);
        return { error: error.message, motions: [] };
    }
});

// Read file content (for mouth_track.json)
ipcMain.handle('read-file', async (event, filePath) => {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        return { content: content };
    } catch (error) {
        console.error('[Main] Error reading file:', filePath, error);
        return { error: error.message };
    }
});

// Load single asset folder (flat structure: video + mouth_track.json + mouth/)
ipcMain.handle('load-single-asset', async (event, folderPath) => {
    console.log('[Main] Loading single asset folder:', folderPath);

    if (!folderPath || !fs.existsSync(folderPath)) {
        return { error: 'Folder not found' };
    }

    try {
        const files = fs.readdirSync(folderPath);

        // Find mouthless video (WebM or MP4)
        let videoFile = files.find(f =>
            f.includes('mouthless') && f.endsWith('.webm')
        );
        if (!videoFile) {
            videoFile = files.find(f =>
                f.includes('mouthless') && f.includes('h264') && f.endsWith('.mp4')
            );
        }
        if (!videoFile) {
            videoFile = files.find(f =>
                f.includes('mouthless') && f.endsWith('.mp4')
            );
        }

        // Find mouth_track.json
        const trackFile = files.find(f => f === 'mouth_track.json');

        // Find mouth folder
        const mouthPath = path.join(folderPath, 'mouth');
        const hasMouth = fs.existsSync(mouthPath) && fs.statSync(mouthPath).isDirectory();

        const missing = [];
        if (!videoFile) missing.push('*_mouthless video');
        if (!trackFile) missing.push('mouth_track.json');
        if (!hasMouth) missing.push('mouth/ folder');

        if (missing.length > 0) {
            return { error: 'Missing: ' + missing.join(', ') };
        }

        // Check required sprites
        const requiredSprites = ['closed.png', 'open.png'];
        for (const sprite of requiredSprites) {
            if (!fs.existsSync(path.join(mouthPath, sprite))) {
                missing.push('mouth/' + sprite);
            }
        }
        if (missing.length > 0) {
            return { error: 'Missing: ' + missing.join(', ') };
        }

        // Get available sprites
        const availableSprites = ['closed.png', 'open.png'];
        for (const sprite of ['half.png', 'e.png', 'u.png']) {
            if (fs.existsSync(path.join(mouthPath, sprite))) {
                availableSprites.push(sprite);
            }
        }

        const result = {
            videoPath: path.join(folderPath, videoFile),
            trackPath: path.join(folderPath, trackFile),
            mouthPath: mouthPath,
            availableSprites: availableSprites
        };

        console.log('[Main] Single asset loaded:', videoFile);
        return result;

    } catch (error) {
        console.error('[Main] Error loading single asset:', error);
        return { error: error.message };
    }
});

// Load playlist JSON for review mode
ipcMain.handle('load-playlist', async (event, jsonPath) => {
    console.log('[Main] Loading playlist:', jsonPath);
    try {
        const data = fs.readFileSync(jsonPath, 'utf8');
        const playlist = JSON.parse(data);
        const motions = [];

        for (const entry of (playlist.videos || [])) {
            if (!entry.videoPath || !entry.trackPath || !entry.mouthPath) continue;
            if (!fs.existsSync(entry.videoPath) || !fs.existsSync(entry.trackPath)) {
                console.warn('[Main] Playlist: missing file for', entry.name);
                continue;
            }
            const mouthPath = entry.mouthPath;
            if (!fs.existsSync(path.join(mouthPath, 'open.png')) ||
                !fs.existsSync(path.join(mouthPath, 'closed.png'))) {
                console.warn('[Main] Playlist: missing mouth sprites for', entry.name);
                continue;
            }
            const availableSprites = ['closed.png', 'open.png'];
            for (const s of ['half.png', 'e.png', 'u.png']) {
                if (fs.existsSync(path.join(mouthPath, s))) {
                    availableSprites.push(s);
                }
            }
            motions.push({
                name: entry.name || path.basename(entry.videoPath, '.webm'),
                videoPath: entry.videoPath,
                trackPath: entry.trackPath,
                mouthPath: mouthPath,
                availableSprites: availableSprites
            });
        }
        console.log('[Main] Playlist: loaded', motions.length, 'motions');
        return { motions };
    } catch (error) {
        console.error('[Main] Playlist load error:', error);
        return { error: error.message, motions: [] };
    }
});

// App lifecycle
app.whenReady().then(() => {
    parseArgs();
    loadSettings();
    registerLocalProtocol();
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    app.quit();
});

// debounce中の未保存設定を終了前にフラッシュ
app.on('before-quit', () => {
    if (saveSettingsTimer) {
        clearTimeout(saveSettingsTimer);
        saveSettingsTimer = null;
        saveSettings();
    }
});
