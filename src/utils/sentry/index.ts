import * as Sentry from '@sentry/node';
import { config } from '../../config';
import { logger } from '../logger';

export interface SentryUser {
  id?: string;
  email?: string;
  ip_address?: string;
}

export interface SentryContext {
  [key: string]: any;
}

export interface SentryTag {
  [key: string]: string | number | boolean;
}

class SentryService {
  private initialized = false;

  async initialize(): Promise<void> {
    if (!config.sentry.enabled) {
      logger.info('Sentry is disabled');
      return;
    }

    if (!config.sentry.dsn) {
      logger.warn('Sentry DSN not configured, skipping initialization');
      return;
    }

    if (this.initialized) {
      logger.warn('Sentry already initialized');
      return;
    }

    try {
      Sentry.init({
        dsn: config.sentry.dsn,
        sendDefaultPii: true,
        environment: config.nodeEnv,
        debug: config.sentry.debug,
        sampleRate: config.sentry.sampleRate,
        tracesSampleRate: config.sentry.tracesSampleRate,
        attachStacktrace: config.sentry.attachStacktrace,
        integrations: [
          Sentry.httpIntegration(),
          Sentry.expressIntegration(),
          Sentry.mongoIntegration(),
          Sentry.onUnhandledRejectionIntegration({
            mode: 'warn',
          }),
          ...(config.sentry.captureConsole
            ? [Sentry.consoleIntegration()]
            : []),
        ],
        beforeSend: event => {
          if (config.nodeEnv === 'development') {
            logger.debug('Sentry event:', event as any);
          }
          return event;
        },
        beforeSendTransaction: event => {
          if (config.nodeEnv === 'development') {
            logger.debug('Sentry transaction:', event as any);
          }
          return event;
        },
      });

      this.initialized = true;
      logger.info('Sentry initialized successfully', {
        environment: config.nodeEnv,
        sampleRate: config.sentry.sampleRate,
        tracesSampleRate: config.sentry.tracesSampleRate,
      });
    } catch (error) {
      logger.error('Failed to initialize Sentry', error);
    }
  }

  captureException(
    error: Error | string | unknown,
    context?: {
      user?: SentryUser;
      tags?: SentryTag;
      contexts?: SentryContext;
      extra?: Record<string, any>;
      level?: Sentry.SeverityLevel;
      fingerprint?: string[];
    }
  ): string | undefined {
    if (!this.initialized || !config.sentry.enabled) {
      return undefined;
    }

    const scope = new Sentry.Scope();

    if (context) {
      if (context.user) {
        scope.setUser(context.user);
      }

      if (context.tags) {
        Object.entries(context.tags).forEach(([key, value]) => {
          scope.setTag(key, value);
        });
      }

      if (context.contexts) {
        Object.entries(context.contexts).forEach(([key, value]) => {
          scope.setContext(key, value);
        });
      }

      if (context.extra) {
        Object.entries(context.extra).forEach(([key, value]) => {
          scope.setExtra(key, value);
        });
      }

      if (context.level) {
        scope.setLevel(context.level);
      }

      if (context.fingerprint) {
        scope.setFingerprint(context.fingerprint);
      }
    }

    const eventId = Sentry.captureException(error, scope);
    return eventId;
  }

  addBreadcrumb(breadcrumb: Sentry.Breadcrumb): void {
    if (!this.initialized || !config.sentry.enabled) {
      return;
    }

    Sentry.addBreadcrumb(breadcrumb);
  }

  startTransaction(
    name: string,
    options?: Record<string, any>
  ): Sentry.Span | undefined {
    if (!this.initialized || !config.sentry.enabled) {
      return undefined;
    }

    return Sentry.startSpan({ name, ...options }, () => {}) as any;
  }

  async flush(timeout?: number): Promise<boolean> {
    if (!this.initialized || !config.sentry.enabled) {
      return true;
    }

    return Sentry.flush(timeout);
  }

  async close(): Promise<boolean> {
    if (!this.initialized || !config.sentry.enabled) {
      return true;
    }

    const result = await Sentry.close();
    this.initialized = false;
    return result;
  }

  configureScope(callback: (scope: Sentry.Scope) => void): void {
    if (!this.initialized || !config.sentry.enabled) {
      return;
    }

    callback(Sentry.getCurrentScope());
  }
  isInitialized(): boolean {
    return this.initialized && config.sentry.enabled;
  }
}
export const sentryService = new SentryService();
export { Sentry };
