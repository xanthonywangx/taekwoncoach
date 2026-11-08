// SpacetimeDB module for Taekwoncoach, following the whiteboard design.
// Publish from the backend folder: spacetime publish taekwoncoach --server local
//
// user          : userId, username (unique, no password)
// kick          : kickId, userId, resultId (links to the Python results/ folder), recordedTime, statusId
// evaluation    : one row per completed kick (the 4 checks, their measurements, tier, feedback, score)
// media_asset   : files for a kick (video, chamber, extension, result); fileName is a URL path on the Python API
// media_data    : the file BYTES for a media_asset (base64), one row per asset. Kept in its own table so
//                 loading kicks/evaluations never pulls videos. Subscribe to it narrowly
//                 (SELECT * FROM media_data WHERE asset_id = X), never the whole table.
// record_status : lookup 1 Completed, 2 Processing, 3 Uploaded, 4 Failed
// file_type     : lookup 1 Video, 2 Chamber, 3 Extension, 4 Result
//
// Media bytes: SpacetimeDB keeps tables in memory, so files are capped at MAX_MEDIA_BYTES each
// (~1 MB demo videos are fine). Python also keeps a copy on disk in results/.
//
// Lifecycle written by the Python backend:
//   start_kick(username, resultId)          -> kick row, status Uploaded (user must already exist)
//   set_kick_status(resultId, statusId)     -> e.g. Processing while grading
//   complete_kick(resultId, resultJson)     -> evaluation + media assets, status Completed or Failed
//   upload_media(resultId, fileTypeId, mimeType, fileData) -> stores one file's bytes (after complete_kick)
// Accounts (username only, no password; hackathon demo):
//   create_user(username)   sign up. Usernames are stored lowercase: 3-20 letters, numbers or _.
//   Log in = the Python API checks the user row exists. start_kick refuses unknown usernames.
// From the frontend:
//   create_user(username), delete_kick(resultId)
//
// Errors meant for the user use SenderError: the HTTP API returns them as status 530 with the
// message as the body. (A plain Error crashes the call with "fatal error" instead.)
import { schema, table, t, SenderError } from 'spacetimedb/server';

const STATUS = { COMPLETED: 1, PROCESSING: 2, UPLOADED: 3, FAILED: 4 } as const;
const FILE_TYPE = { VIDEO: 1, CHAMBER: 2, EXTENSION: 3, RESULT: 4 } as const;

const spacetimedb = schema({
  user: table(
    { public: true },
    {
      userId: t.u64().primaryKey().autoInc(),
      username: t.string().unique(),
      createdAt: t.timestamp(),
    }
  ),

  kick: table(
    { public: true },
    {
      kickId: t.u64().primaryKey().autoInc(),
      userId: t.u64().index('btree'),
      resultId: t.string().unique(),
      recordedTime: t.timestamp(),
      statusId: t.u8(),
      failReason: t.string(),          // why a kick Failed (e.g. "Step back so head to feet are in frame")
    }
  ),

  evaluation: table(
    { public: true },
    {
      kickId: t.u64().primaryKey(),
      chamberHeightResult: t.bool(),
      kneeBelowHipRatio: t.f64(),
      chamberTightnessResult: t.bool(),
      ankleForwardKneeRatio: t.f64(),
      kickAngleResult: t.bool(),
      legAngle: t.f64(),
      kickStraightnessResult: t.bool(),
      kneeAngle: t.f64(),
      kickTier: t.string(),            // strong | even | building
      feedback: t.string(),
      totalScore: t.u8(),              // 0-4
      feedbackSource: t.string(),      // nemotron | fallback
    }
  ),

  mediaAsset: table(
    { public: true },
    {
      assetId: t.u64().primaryKey().autoInc(),
      kickId: t.u64().index('btree'),
      fileTypeId: t.u8(),
      fileName: t.string(),            // e.g. /results/<resultId>/chamber.jpg (prefix with the API URL)
    }
  ),

  mediaData: table(
    { public: true },
    {
      assetId: t.u64().primaryKey(),   // same id as the media_asset row it belongs to
      kickId: t.u64().index('btree'),
      mimeType: t.string(),            // video/webm, video/mp4, image/jpeg
      sizeBytes: t.u32(),              // decoded size
      fileData: t.string(),            // the file, base64-encoded
    }
  ),

  recordStatus: table(
    { public: true },
    {
      statusId: t.u8().primaryKey(),
      status: t.string(),
    }
  ),

  fileType: table(
    { public: true },
    {
      fileTypeId: t.u8().primaryKey(),
      fileType: t.string(),
    }
  ),
});
export default spacetimedb;

const USERNAME_RE = /^[a-z0-9_]{3,20}$/;

function cleanUsername(raw: string): string {
  const name = raw.trim().toLowerCase();
  if (!USERNAME_RE.test(name)) {
    throw new SenderError('Username must be 3-20 characters: letters, numbers, or _');
  }
  return name;
}

const MAX_MEDIA_BYTES = 8 * 1024 * 1024;   // per file; demo clips are ~1 MB

const STATUS_ROWS: [number, string][] = [[1, 'Completed'], [2, 'Processing'], [3, 'Uploaded'], [4, 'Failed']];
const FILE_TYPE_ROWS: [number, string][] = [[1, 'Video'], [2, 'Chamber'], [3, 'Extension'], [4, 'Result']];

// Fills the two lookup tables. Runs on first publish (init) and again from start_kick if they're
// empty, e.g. when the module was republished onto an existing database without clearing it.
export const init = spacetimedb.init(ctx => {
  for (const [statusId, status] of STATUS_ROWS) ctx.db.recordStatus.insert({ statusId, status });
  for (const [fileTypeId, fileType] of FILE_TYPE_ROWS) ctx.db.fileType.insert({ fileTypeId, fileType });
});

export const createUser = spacetimedb.reducer(
  { username: t.string() },
  (ctx, { username }) => {
    const name = cleanUsername(username);
    if (ctx.db.user.username.find(name)) throw new SenderError(`Username "${name}" is taken`);
    ctx.db.user.insert({ userId: 0n, username: name, createdAt: ctx.timestamp });
  }
);

export const startKick = spacetimedb.reducer(
  { username: t.string(), resultId: t.string() },
  (ctx, { username, resultId }) => {
    const name = cleanUsername(username);
    const user = ctx.db.user.username.find(name);
    if (!user) throw new SenderError(`No account named "${name}". Sign up first.`);
    if (ctx.db.kick.resultId.find(resultId)) throw new SenderError(`Kick ${resultId} already exists`);
    for (const [statusId, status] of STATUS_ROWS) {
      if (!ctx.db.recordStatus.statusId.find(statusId)) ctx.db.recordStatus.insert({ statusId, status });
    }
    for (const [fileTypeId, fileType] of FILE_TYPE_ROWS) {
      if (!ctx.db.fileType.fileTypeId.find(fileTypeId)) ctx.db.fileType.insert({ fileTypeId, fileType });
    }
    ctx.db.kick.insert({
      kickId: 0n,
      userId: user.userId,
      resultId,
      recordedTime: ctx.timestamp,
      statusId: STATUS.UPLOADED,
      failReason: '',
    });
  }
);

export const setKickStatus = spacetimedb.reducer(
  { resultId: t.string(), statusId: t.u8() },
  (ctx, { resultId, statusId }) => {
    const kick = ctx.db.kick.resultId.find(resultId);
    if (!kick) throw new SenderError(`No kick ${resultId}`);
    if (!ctx.db.recordStatus.statusId.find(statusId)) throw new SenderError(`Unknown status ${statusId}`);
    kick.statusId = statusId;
    ctx.db.kick.kickId.update(kick);
  }
);

export const completeKick = spacetimedb.reducer(
  { resultId: t.string(), resultJson: t.string() },
  (ctx, { resultId, resultJson }) => {
    const kick = ctx.db.kick.resultId.find(resultId);
    if (!kick) throw new SenderError(`No kick ${resultId}; call start_kick first`);
    const r = JSON.parse(resultJson);
    const num = (v: unknown) => (typeof v === 'number' && isFinite(v) ? v : 0);
    const str = (v: unknown) => (typeof v === 'string' ? v : '');

    // Media assets (replace any from an earlier attempt, including uploaded bytes)
    ctx.db.mediaData.kickId.delete(kick.kickId);
    ctx.db.mediaAsset.kickId.delete(kick.kickId);
    const files: [number, string][] = [
      [FILE_TYPE.VIDEO, str(r.clip_url)],
      [FILE_TYPE.CHAMBER, str(r.images?.chamber)],
      [FILE_TYPE.EXTENSION, str(r.images?.extension)],
      [FILE_TYPE.RESULT, `/results/${resultId}`],
    ];
    for (const [fileTypeId, fileName] of files) {
      if (fileName) ctx.db.mediaAsset.insert({ assetId: 0n, kickId: kick.kickId, fileTypeId, fileName });
    }

    if (r.status !== 'graded') {
      kick.statusId = STATUS.FAILED;
      kick.failReason = str(r.retake_reason) || 'Could not grade this kick';
      ctx.db.kick.kickId.update(kick);
      return;
    }

    const c = r.checks ?? {};
    const pass = (k: string) => c[k]?.pass === 1;
    const evaluation = {
      kickId: kick.kickId,
      chamberHeightResult: pass('chamber_height'),
      kneeBelowHipRatio: num(c.chamber_height?.knee_below_hip_ratio),
      chamberTightnessResult: pass('chamber_tightness'),
      ankleForwardKneeRatio: num(c.chamber_tightness?.ankle_forward_of_knee_ratio),
      kickAngleResult: pass('kick_angle'),
      legAngle: num(c.kick_angle?.leg_angle_deg),
      kickStraightnessResult: pass('kick_straightness'),
      kneeAngle: num(c.kick_straightness?.knee_angle_deg),
      kickTier: str(r.tier),
      feedback: str(r.feedback),
      totalScore: num(r.passed),
      feedbackSource: str(r.feedback_source),
    };
    if (ctx.db.evaluation.kickId.find(kick.kickId)) ctx.db.evaluation.kickId.update(evaluation);
    else ctx.db.evaluation.insert(evaluation);

    kick.statusId = STATUS.COMPLETED;
    kick.failReason = '';
    ctx.db.kick.kickId.update(kick);
  }
);

export const deleteKick = spacetimedb.reducer(
  { resultId: t.string() },
  (ctx, { resultId }) => {
    const kick = ctx.db.kick.resultId.find(resultId);
    if (!kick) return;
    ctx.db.mediaData.kickId.delete(kick.kickId);
    ctx.db.mediaAsset.kickId.delete(kick.kickId);
    ctx.db.evaluation.kickId.delete(kick.kickId);
    ctx.db.kick.resultId.delete(resultId);
  }
);

export const uploadMedia = spacetimedb.reducer(
  { resultId: t.string(), fileTypeId: t.u8(), mimeType: t.string(), fileData: t.string() },
  (ctx, { resultId, fileTypeId, mimeType, fileData }) => {
    const kick = ctx.db.kick.resultId.find(resultId);
    if (!kick) throw new SenderError(`No kick ${resultId}; call start_kick first`);
    if (!ctx.db.fileType.fileTypeId.find(fileTypeId)) throw new SenderError(`Unknown file type ${fileTypeId}`);
    if (!fileData) throw new SenderError('Empty file');
    const pad = fileData.endsWith('==') ? 2 : fileData.endsWith('=') ? 1 : 0;
    const sizeBytes = Math.floor((fileData.length * 3) / 4) - pad;
    if (sizeBytes > MAX_MEDIA_BYTES) {
      throw new SenderError(`File is ${(sizeBytes / 1048576).toFixed(1)} MB; the limit is ${MAX_MEDIA_BYTES / 1048576} MB`);
    }

    // One media_asset per (kick, file type): reuse it if complete_kick already made one
    let asset = [...ctx.db.mediaAsset.kickId.filter(kick.kickId)].find(a => a.fileTypeId === fileTypeId);
    if (!asset) {
      const path = fileTypeId === FILE_TYPE.VIDEO ? 'clip'
        : fileTypeId === FILE_TYPE.CHAMBER ? 'chamber.jpg'
        : fileTypeId === FILE_TYPE.EXTENSION ? 'extension.jpg' : 'result';
      asset = ctx.db.mediaAsset.insert({ assetId: 0n, kickId: kick.kickId, fileTypeId, fileName: `/results/${resultId}/${path}` });
    }
    const row = { assetId: asset.assetId, kickId: kick.kickId, mimeType, sizeBytes, fileData };
    if (ctx.db.mediaData.assetId.find(asset.assetId)) ctx.db.mediaData.assetId.update(row);
    else ctx.db.mediaData.insert(row);
  }
);
