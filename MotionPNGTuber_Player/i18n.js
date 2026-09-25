/**
 * i18n.js - Minimal JA/EN localization for MotionPNGTuber Player
 *
 * Plain script (no modules). Must be loaded BEFORE the inline script of
 * index-electron.html (lipsync.js itself is shared with AG and does not use it).
 * Exposes: window.t, window.applyI18n, window.setLanguage, window.i18nGetLanguage
 *
 * Language priority: URL query (?lang=ja|en) > localStorage('mptp_lang') > navigator.language
 */
(function () {
    'use strict';

    var I18N_DICT = {
        ja: {
            // --- index-electron.html: AG と同一の lipsync.js（英語の生ログ）に被せる見出し ---
            'player.load_failed': '読み込みに失敗しました: {message}',
            'player.engine_error': 'プレイヤーエラー: {message}'
        },
        en: {
            // --- index-electron.html: headline over lipsync.js raw English messages ---
            'player.load_failed': 'Load failed: {message}',
            'player.engine_error': 'Player error: {message}'
        }
    };

    function detectLanguage() {
        // 1. URL query: ?lang=ja|en
        try {
            var q = new URLSearchParams(window.location.search).get('lang');
            if (q === 'ja' || q === 'en') return q;
        } catch (e) { /* ignore */ }
        // 2. localStorage
        try {
            var stored = localStorage.getItem('mptp_lang');
            if (stored === 'ja' || stored === 'en') return stored;
        } catch (e) { /* ignore */ }
        // 3. navigator.language (ja* -> ja, otherwise en)
        var nav = (navigator.language || 'en').toLowerCase();
        return nav.indexOf('ja') === 0 ? 'ja' : 'en';
    }

    var currentLang = detectLanguage();

    function t(key, params) {
        var text;
        var dict = I18N_DICT[currentLang];
        if (dict && Object.prototype.hasOwnProperty.call(dict, key)) {
            text = dict[key];
        } else if (Object.prototype.hasOwnProperty.call(I18N_DICT.en, key)) {
            text = I18N_DICT.en[key];
        } else {
            text = key;
        }
        if (params) {
            text = text.replace(/\{(\w+)\}/g, function (match, name) {
                return Object.prototype.hasOwnProperty.call(params, name)
                    ? String(params[name])
                    : match;
            });
        }
        return text;
    }

    function applyI18n() {
        document.documentElement.lang = currentLang;
        var nodes = document.querySelectorAll('[data-i18n]');
        for (var i = 0; i < nodes.length; i++) {
            nodes[i].textContent = t(nodes[i].getAttribute('data-i18n'));
        }
    }

    function setLanguage(lang) {
        if (lang !== 'ja' && lang !== 'en') return;
        try {
            localStorage.setItem('mptp_lang', lang);
        } catch (e) { /* ignore */ }
        location.reload();
    }

    function i18nGetLanguage() {
        return currentLang;
    }

    window.t = t;
    window.applyI18n = applyI18n;
    window.setLanguage = setLanguage;
    window.i18nGetLanguage = i18nGetLanguage;
})();
