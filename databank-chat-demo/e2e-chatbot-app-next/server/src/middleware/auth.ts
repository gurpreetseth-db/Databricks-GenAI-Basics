import type { Request, Response, NextFunction } from 'express';
import { randomUUID } from 'node:crypto';
import { getAuthSession, type AuthSession } from '@chat-template/auth';
import { checkChatAccess } from '@chat-template/core';
import { ChatSDKError } from '@chat-template/core/errors';

/**
 * Short-term memory mode: scope chat ownership/history to a per-browser session
 * so history resets when the app is closed and reopened. Backed by a session
 * cookie (no Max-Age → the browser drops it on close). Within a session the
 * cookie survives reloads, so history accumulates as expected.
 */
function parseCookies(header?: string): Record<string, string> {
  const out: Record<string, string> = {};
  if (!header) return out;
  for (const part of header.split(';')) {
    const idx = part.indexOf('=');
    if (idx > -1) {
      out[part.slice(0, idx).trim()] = decodeURIComponent(part.slice(idx + 1).trim());
    }
  }
  return out;
}

function getOrSetSessionToken(req: Request, res: Response): string {
  const existing = parseCookies(req.headers.cookie)['db_session'];
  if (existing) return existing;
  const token = randomUUID();
  res.append('Set-Cookie', `db_session=${token}; Path=/; HttpOnly; SameSite=Lax`);
  return token;
}

// Extend Express Request type to include session
declare global {
  namespace Express {
    interface Request {
      session?: AuthSession;
    }
  }
}

/**
 * Middleware to authenticate requests and attach session to request object
 */
export async function authMiddleware(
  req: Request,
  res: Response,
  next: NextFunction,
) {
  try {
    const session = await getAuthSession({
      getRequestHeader: (name: string) =>
        req.headers[name.toLowerCase()] as string | null,
    });
    // In short-term mode, namespace the user id with a per-browser session
    // token so history is session-scoped and resets on reopen. Long-term and
    // simple modes leave the id untouched (real per-user id / ephemeral).
    if (session?.user && process.env.MEMORY_MODE === 'shortterm') {
      const token = getOrSetSessionToken(req, res);
      session.user.id = `${session.user.id}::s::${token}`;
    }
    req.session = session || undefined;
    next();
  } catch (error) {
    console.error('Auth middleware error:', error);
    next(error);
  }
}

/**
 * Middleware to require authentication - returns 401 if no session
 */
export function requireAuth(req: Request, res: Response, next: NextFunction) {
  if (!req.session?.user) {
    const response = new ChatSDKError('unauthorized:chat').toResponse();
    return res.status(response.status).json(response.json);
  }
  next();
}

export async function requireChatAccess(
  req: Request,
  res: Response,
  next: NextFunction,
) {
  const id = getIdFromRequest(req);
  if (!id) {
    console.error(
      'Chat access middleware error: no chat ID provided',
      req.params,
    );
    const error = new ChatSDKError('bad_request:api');
    const response = error.toResponse();
    return res.status(response.status).json(response.json);
  }
  const { allowed, reason } = await checkChatAccess(id, req.session?.user.id);
  if (!allowed) {
    console.error(
      'Chat access middleware error: user does not have access to chat',
      reason,
    );
    const error = new ChatSDKError('forbidden:chat', reason);
    const response = error.toResponse();
    return res.status(response.status).json(response.json);
  }
  next();
}

export const getIdFromRequest = (req: Request): string | undefined => {
  const { id } = req.params;
  if (!id) return undefined;
  return typeof id === 'string' ? id : id[0];
};
