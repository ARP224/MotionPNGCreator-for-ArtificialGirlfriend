/**
 * Multi-Motion Manager
 *
 * Manages multiple motion videos for a character,
 * handling random selection and seamless transitions
 * between motions using double-buffering technique.
 */
class MultiMotionManager {
    /**
     * @param {LipsyncEngine} lipsyncEngine - LipsyncEngine instance
     */
    constructor(lipsyncEngine) {
        this.lipsyncEngine = lipsyncEngine;

        // Motion data
        this.motions = [];
        this.currentIndex = -1;
        this.nextIndex = -1;

        // Video elements (double buffer for seamless transition)
        this.frontVideo = null;  // Currently displayed video
        this.backVideo = null;   // Preloading video

        // Preload state
        this.nextMotionReady = false;
        this.preloadedSprites = {};
        this.preloadedSpriteUrls = {};
        this._preloadedMotionName = null;  // preloadedSprites がどのモーションのものか
        this._preloadGen = 0;              // プリロード世代（古い完了を無効化）
        this._preloadListener = null;      // 残留 canplaythrough リスナー解除用
        this._preloadTarget = null;
        this._gotoInProgress = false;      // goto実行中（多重goto・自動遷移との競合防止）

        // State
        this.isPlaying = false;
        this.isTransitioning = false;

        // Playback mode: 'random' (default) or 'sequential'
        this.playbackMode = 'random';

        // Callback: (name: string) => void  — called after every motion change
        this.onMotionChanged = null;
    }

    /**
     * Initialize with motion data
     * @param {Array} motionDataList - List of motion data from main process
     */
    async init(motionDataList) {
        console.log('[MultiMotion] Initializing with', motionDataList.length, 'motions');

        // Initialize double buffer video elements
        this.frontVideo = this.lipsyncEngine.video;  // #base-video
        this.backVideo = document.getElementById('back-video');

        // Configure back video element
        this.backVideo.muted = true;
        this.backVideo.playsInline = true;
        this.backVideo.preload = 'auto';
        this.backVideo.loop = false;

        // Load all motion data
        for (const motion of motionDataList) {
            try {
                // Read mouth_track.json
                const result = await window.electronAPI.readFile(motion.trackPath);
                if (result.error) {
                    console.error('[MultiMotion] Failed to read track data:', motion.name, result.error);
                    continue;
                }

                const trackData = JSON.parse(result.content);

                // Build sprite paths
                const spritePaths = {};
                for (const sprite of motion.availableSprites) {
                    const spriteName = sprite.replace('.png', '');
                    spritePaths[spriteName] = this._buildFilePath(motion.mouthPath, sprite);
                }

                this.motions.push({
                    name: motion.name,
                    videoPath: this._buildFilePath(motion.videoPath),
                    trackData: trackData,
                    mouthPath: motion.mouthPath,
                    spritePaths: spritePaths
                });

                console.log('[MultiMotion] Loaded motion:', motion.name);

            } catch (error) {
                console.error('[MultiMotion] Error loading motion:', motion.name, error);
            }
        }

        if (this.motions.length === 0) {
            throw new Error('No valid motions loaded');
        }

        console.log('[MultiMotion] Total valid motions:', this.motions.length);

        // Load first motion (uses traditional method for initial load)
        await this._switchMotion(0);
    }

    /**
     * Build file path with local:// protocol
     */
    _buildFilePath(filePath, fileName = null) {
        let fullPath = fileName ? `${filePath}\\${fileName}` : filePath;
        // Normalize path separators
        fullPath = fullPath.replace(/\\/g, '/');
        return `local://${fullPath}`;
    }

    /**
     * Get current track data
     */
    getCurrentTrackData() {
        if (this.currentIndex >= 0 && this.currentIndex < this.motions.length) {
            return this.motions[this.currentIndex].trackData;
        }
        return null;
    }

    /**
     * Switch to a specific motion (used for initial load only)
     * @param {number} index - Motion index
     */
    async _switchMotion(index) {
        if (index < 0 || index >= this.motions.length) {
            console.error('[MultiMotion] Invalid motion index:', index);
            return;
        }

        const motion = this.motions[index];
        console.log('[MultiMotion] Switching to motion:', motion.name);

        this.currentIndex = index;
        this.isTransitioning = true;

        try {
            // Update video source
            this.frontVideo.src = motion.videoPath;
            this.frontVideo.loop = false;  // Don't loop - we handle transitions

            // Update LipsyncEngine track data
            this.lipsyncEngine.trackData = motion.trackData;

            // Load mouth sprites
            await this.lipsyncEngine._loadMouthSprites(motion.spritePaths);

            // Initialize mouth state to closed
            this.lipsyncEngine.setMouthState('closed', true);

            // Update canvas size
            if (motion.trackData.width && motion.trackData.height) {
                this.lipsyncEngine.mouthCanvas.width = motion.trackData.width;
                this.lipsyncEngine.mouthCanvas.height = motion.trackData.height;
            }

            // Set up video end handler for transition
            this.frontVideo.onended = () => {
                if (this.isPlaying && !this._gotoInProgress) {
                    this._transitionToNextMotion();
                }
            };

            // Wait for video to be ready
            await new Promise((resolve, reject) => {
                const onReady = () => {
                    this.frontVideo.removeEventListener('canplaythrough', onReady);
                    this.frontVideo.removeEventListener('loadeddata', onReady);
                    resolve();
                };
                const onError = (e) => {
                    reject(new Error('Video load failed: ' + (e.message || 'unknown')));
                };
                this.frontVideo.addEventListener('canplaythrough', onReady);
                this.frontVideo.addEventListener('loadeddata', onReady);
                this.frontVideo.onerror = onError;
                this.frontVideo.load();
            });

            // Preload next motion
            this._preloadNextMotion();

            console.log('[MultiMotion] Motion switch complete:', motion.name);

        } catch (error) {
            console.error('[MultiMotion] Error switching motion:', error);
        } finally {
            this.isTransitioning = false;
        }
    }

    /**
     * Preload the next random motion into backVideo
     */
    _preloadNextMotion() {
        // Select next motion index
        if (this.motions.length <= 1) {
            this.nextIndex = 0;
        } else if (this.playbackMode === 'sequential') {
            this.nextIndex = (this.currentIndex + 1) % this.motions.length;
        } else {
            // Select random motion (different from current)
            do {
                this.nextIndex = Math.floor(Math.random() * this.motions.length);
            } while (this.nextIndex === this.currentIndex);
        }

        const nextMotion = this.motions[this.nextIndex];
        this.nextMotionReady = false;

        // 前回のプリロードを無効化（残留リスナー解除 + 世代更新）
        this._cancelPendingPreload();
        const gen = this._preloadGen;

        console.log('[MultiMotion] Preloading next motion:', nextMotion.name);

        // Preload video into backVideo
        this.backVideo.src = nextMotion.videoPath;
        this.backVideo.load();

        // Wait for video to be ready, then preload sprites
        const onVideoReady = () => {
            this.backVideo.removeEventListener('canplaythrough', onVideoReady);
            if (gen !== this._preloadGen) return;  // 古い世代の完了は無視

            // Preload mouth sprites
            this._preloadSprites(nextMotion.spritePaths).then(({ sprites, spriteUrls }) => {
                if (gen !== this._preloadGen) return;
                this.preloadedSprites = sprites;
                this.preloadedSpriteUrls = spriteUrls;
                this._preloadedMotionName = nextMotion.name;
                this.nextMotionReady = true;
                console.log('[MultiMotion] Next motion fully preloaded:', nextMotion.name);
            }).catch(error => {
                if (gen !== this._preloadGen) return;
                console.error('[MultiMotion] Failed to preload sprites:', error);
                // Mark as ready anyway to prevent infinite waiting
                // （スプライトは遷移時に spritePaths から直接ロードされる）
                this.nextMotionReady = true;
            });
        };

        this._preloadListener = onVideoReady;
        this._preloadTarget = this.backVideo;
        this.backVideo.addEventListener('canplaythrough', onVideoReady);
    }

    /**
     * Invalidate any in-flight preload (listener + generation)
     */
    _cancelPendingPreload() {
        this._preloadGen++;
        if (this._preloadTarget && this._preloadListener) {
            this._preloadTarget.removeEventListener('canplaythrough', this._preloadListener);
        }
        this._preloadListener = null;
        this._preloadTarget = null;
        this._preloadedMotionName = null;
    }

    /**
     * Preload mouth sprites for the next motion
     * @param {Object} spritePaths - Map of sprite name to path
     * @returns {Promise<{sprites: Object, spriteUrls: Object}>}
     */
    async _preloadSprites(spritePaths) {
        const sprites = {};
        const spriteUrls = {};

        const promises = Object.entries(spritePaths).map(async ([key, src]) => {
            const img = new Image();
            img.src = src;
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = () => reject(new Error(`Failed to load sprite: ${key}`));
            });
            sprites[key] = img;
            spriteUrls[key] = src;
        });

        await Promise.all(promises);
        return { sprites, spriteUrls };
    }

    /**
     * Transition to the next motion using double-buffering
     * This is called when the current video ends
     */
    async _transitionToNextMotion() {
        if (this.isTransitioning) {
            console.log('[MultiMotion] Already transitioning, skipping');
            return;
        }

        this.isTransitioning = true;
        console.log('[MultiMotion] Transitioning to next motion');

        const nextMotion = this.motions[this.nextIndex];

        try {
            // Wait for preload to complete (with timeout)
            if (!this.nextMotionReady) {
                console.log('[MultiMotion] Waiting for preload to complete...');
                const timeoutMs = 5000;
                const startTime = Date.now();

                await new Promise((resolve) => {
                    const check = () => {
                        if (this.nextMotionReady) {
                            resolve();
                        } else if (Date.now() - startTime > timeoutMs) {
                            console.warn('[MultiMotion] Preload timeout, proceeding anyway');
                            resolve();
                        } else {
                            setTimeout(check, 50);
                        }
                    };
                    check();
                });
            }

            // 1. Update LipsyncEngine settings before switching video
            this.lipsyncEngine.trackData = nextMotion.trackData;
            if (this._preloadedMotionName === nextMotion.name) {
                this.lipsyncEngine.mouthSprites = this.preloadedSprites;
                this.lipsyncEngine.mouthSpriteUrls = this.preloadedSpriteUrls;
            } else {
                // プリロード未完了/不一致: 別モーションのスプライト混入を防ぐため直接ロード
                console.warn('[MultiMotion] Preloaded sprites unavailable, loading directly:', nextMotion.name);
                try {
                    const { sprites, spriteUrls } = await this._preloadSprites(nextMotion.spritePaths);
                    this.lipsyncEngine.mouthSprites = sprites;
                    this.lipsyncEngine.mouthSpriteUrls = spriteUrls;
                } catch (spriteError) {
                    // ロード失敗時は現行スプライトを維持（描画停止よりまし）
                    console.error('[MultiMotion] Direct sprite load failed:', spriteError);
                }
            }
            this.lipsyncEngine.setMouthState('closed', true);

            // 2. Update canvas size
            if (nextMotion.trackData.width && nextMotion.trackData.height) {
                this.lipsyncEngine.mouthCanvas.width = nextMotion.trackData.width;
                this.lipsyncEngine.mouthCanvas.height = nextMotion.trackData.height;
            }

            // 3. Update video reference in LipsyncEngine
            this.lipsyncEngine.video = this.backVideo;

            // 4. Start playback on backVideo
            this.backVideo.currentTime = 0;
            try {
                await this.backVideo.play();
            } catch (e) {
                if (e.name !== 'AbortError') throw e;
                console.log('[MultiMotion] Transition play aborted');
            }

            // 5. Swap z-index and visibility (bring backVideo to front)
            this.backVideo.style.visibility = 'visible';
            this.backVideo.style.zIndex = '1';
            this.frontVideo.style.zIndex = '0';
            this.frontVideo.style.visibility = 'hidden';

            // 6. Restart render loop for new video
            this.lipsyncEngine.startRenderLoop();

            // 7. Stop old frontVideo and clear its event handler
            this.frontVideo.pause();
            this.frontVideo.onended = null;

            // 8. Set up onended handler for new front video (currently backVideo)
            this.backVideo.onended = () => {
                if (this.isPlaying && !this._gotoInProgress) {
                    this._transitionToNextMotion();
                }
            };

            // 9. Swap front and back references
            [this.frontVideo, this.backVideo] = [this.backVideo, this.frontVideo];
            this.currentIndex = this.nextIndex;

            console.log('[MultiMotion] Transition complete:', nextMotion.name);

            // 10. Fire callback
            if (this.onMotionChanged) {
                this.onMotionChanged(nextMotion.name);
            }

            // 11. Start preloading next motion
            this._preloadNextMotion();

        } catch (error) {
            console.error('[MultiMotion] Error during transition:', error);
        } finally {
            this.isTransitioning = false;
        }
    }

    /**
     * Start playing
     */
    startPlaying() {
        if (this.isPlaying) return;

        this.isPlaying = true;
        this._startVideoPlayback();

        // Start LipsyncEngine render loop
        this.lipsyncEngine.start();

        console.log('[MultiMotion] Started playing');
    }

    /**
     * Stop playing
     */
    stopPlaying() {
        this.isPlaying = false;

        if (this.frontVideo) {
            this.frontVideo.pause();
        }
        if (this.backVideo) {
            this.backVideo.pause();
        }

        this.lipsyncEngine.stop();

        console.log('[MultiMotion] Stopped playing');
    }

    /**
     * Start video playback
     */
    _startVideoPlayback() {
        if (!this.frontVideo) return;

        this.frontVideo.currentTime = 0;
        this.frontVideo.play().catch(error => {
            if (error.name === 'AbortError') {
                // Expected when video source changes during play - ignore
                console.log('[MultiMotion] Play aborted (source change)');
                return;
            }
            console.error('[MultiMotion] Video play error:', error);
            const retryPlay = () => {
                this.frontVideo.play();
                document.removeEventListener('click', retryPlay);
            };
            document.addEventListener('click', retryPlay);
        });
    }

    /**
     * Get current motion name
     */
    getCurrentMotionName() {
        if (this.currentIndex >= 0 && this.currentIndex < this.motions.length) {
            return this.motions[this.currentIndex].name;
        }
        return null;
    }

    /**
     * Remove a motion from the playlist by name (review mode).
     *
     * 削除対象がbackVideoにプリロード済みの場合はsrcを解放してから外す
     * （Python側がファイルをmiss_dataへ移動できるよう、OSのファイル
     * ハンドルを確実に手放す）。現在再生中・遷移中は拒否する。
     *
     * @param {string} name - Motion name
     * @returns {{ok: boolean, removed: boolean, reason: string}}
     */
    removeMotionByName(name) {
        const index = this.motions.findIndex(m => m.name === name);
        if (index < 0) {
            // プレイリストに無い（jsonなし動画等）→ ロックは保持していない
            return { ok: true, removed: false, reason: 'not_found' };
        }
        if (index === this.currentIndex) {
            return { ok: false, removed: false, reason: 'playing' };
        }
        if (this.isTransitioning || this._gotoInProgress) {
            return { ok: false, removed: false, reason: 'busy' };
        }
        if (this.motions.length <= 1) {
            return { ok: false, removed: false, reason: 'last' };
        }

        // プリロード中/済みならbackVideoのファイルハンドルを解放
        this._cancelPendingPreload();
        this.nextMotionReady = false;
        try {
            this.backVideo.pause();
            this.backVideo.removeAttribute('src');
            this.backVideo.load();
        } catch (e) {
            console.warn('[MultiMotion] backVideo release failed:', e);
        }

        this.motions.splice(index, 1);
        if (index < this.currentIndex) {
            this.currentIndex--;
        }
        console.log('[MultiMotion] Removed motion:', name,
                    '(remaining:', this.motions.length + ')');

        // nextIndexを再計算してプリロードし直す
        this._preloadNextMotion();
        return { ok: true, removed: true, reason: '' };
    }

    /**
     * Manually switch to a specific motion by name (uses double-buffer)
     * @param {string} name - Motion name
     * @returns {Promise<boolean>} 切替を実行できたか（見つからない/遷移中は false）
     */
    async switchToMotionByName(name) {
        const index = this.motions.findIndex(m => m.name === name);
        if (index < 0) {
            console.error('[MultiMotion] Motion not found:', name);
            return false;
        }
        if (index === this.currentIndex) return true;
        if (this.isTransitioning || this._gotoInProgress) return false;

        // goto実行中は自動遷移(onended)と後続gotoをブロックして直列化する
        // （awaitの合間に割り込まれると backVideo/preloadedSprites が競合する）
        this._gotoInProgress = true;
        try {
            // Force-preload the target into backVideo, then transition
            // 進行中のプリロードを無効化してから上書きする
            this._cancelPendingPreload();
            this.nextIndex = index;
            this.nextMotionReady = false;

            const targetMotion = this.motions[index];
            console.log('[MultiMotion] Goto:', targetMotion.name);

            // Preload target into backVideo
            this.backVideo.src = targetMotion.videoPath;
            this.backVideo.load();

            await new Promise((resolve) => {
                const onReady = () => {
                    clearTimeout(timer);
                    this.backVideo.removeEventListener('canplaythrough', onReady);
                    resolve();
                };
                // Timeout fallback（リスナーも残さない）
                const timer = setTimeout(() => {
                    this.backVideo.removeEventListener('canplaythrough', onReady);
                    resolve();
                }, 3000);
                this.backVideo.addEventListener('canplaythrough', onReady);
            });

            // Preload sprites
            const loaded = await this._preloadSprites(targetMotion.spritePaths).catch(() => null);
            if (loaded) {
                this.preloadedSprites = loaded.sprites;
                this.preloadedSpriteUrls = loaded.spriteUrls;
                this._preloadedMotionName = targetMotion.name;
            }
            this.nextMotionReady = true;

            // Now do the transition
            await this._transitionToNextMotion();
            return true;
        } finally {
            this._gotoInProgress = false;
        }
    }
}
