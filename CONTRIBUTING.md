<p align="right">
  <a href="./CONTRIBUTING.ja.md">日本語</a> | <strong>English</strong>
</p>

# Contributing

Thanks for your interest in MotionPNGCreator for ArtificialGirlfriend (MPC). Bug reports, feature requests and pull requests are all welcome — **in English or Japanese, whichever you prefer.**

This is a personal project, so replies and reviews may take a while. Thanks for your patience.

## Where to report and discuss

- **[Issues](https://github.com/ARP224/MotionPNGCreator-for-ArtificialGirlfriend/issues)** — bug reports and feature requests. Templates are provided, but the headings are only a guide; write freely and fill in what you can
- **[Discussions](https://github.com/ARP224/MotionPNGCreator-for-ArtificialGirlfriend/discussions)** — usage questions, casual chat and ideas

For bug reports, including your OS and GPU, which step you were on (Step 1 tab / Step 2 page), what you did and what happened, plus the console output from launching with `uv run python asset_preparer.py` (or `uv run python video_generator.py`) speeds up the investigation a lot. **Before pasting screenshots or logs, make sure your Vidu API key is not visible.**

## Pull request policy

- **Small fixes (typos, obvious bug fixes) can be sent straight away**
- **For larger changes, please open an issue first.** Working in a direction that doesn't fit the project wastes your time as well as ours
- `main` is the distribution channel of this repository. Users receive `main` directly via `git pull`, so **merging is releasing**. Changes are merged carefully
- Changes that touch behavior are verified by the maintainer actually running the GUI. Verifying changes that touch video generation involves Vidu credits and GPU processing, so it can take time. **Adding a short "do this, and this should happen" verification recipe to your PR speeds it up**

## Before opening a pull request

Your development environment is simply what the README's setup steps create — no extra setup is needed. There is no automated test suite, so with that in place:

- Actually run the GUI and exercise the steps you touched (Step 1's ①②③⑤⑥⑦ can be verified without spending credits; ④ and Step 2 generation consume Vidu credits)
- If you change the UI text (`locales/`), check the EN/JA consistency: `uv run python tools/check_locales.py`

## Code conventions

Keeping to these four rules is all that's needed.

### 1. Never hard-code UI strings

Write user-facing text as `tr("key")` and add the key to **both** `locales/ja.json` and `locales/en.json`. Consistency can be checked with `uv run python tools/check_locales.py`.

### 2. Anything that spends money goes through "estimate → confirm"

Code that calls the Vidu API must follow the existing pattern: show the video count and estimated credits, then ask for confirmation before starting. Changes that let billing run silently will not be merged.

### 3. Keep .bat / .vbs files CRLF and ASCII-only

The launch scripts require CRLF line endings (a cmd constraint) and are written in ASCII only to avoid mojibake (see `.gitattributes` and the NOTE at the top of each file). Watch out for editors that convert them automatically.

### 4. Don't mix structural and behavioral changes

Keep file moves, renames and splits in separate commits from changes that alter behavior. When they are mixed, it becomes impossible to trace later which commit changed the behavior.

## License of contributions

By submitting a pull request you agree that your contribution is licensed under **AGPL-3.0-only** (GNU Affero General Public License version 3; see `LICENSE`), the same license as the rest of this project. You keep the copyright to your own work. There is no Contributor License Agreement to sign.

The whole repository (including `MotionPNGTuber_Player/`) carries this single license. See [NOTICE.md](NOTICE.md) for per-file origins and original copyright notices.
