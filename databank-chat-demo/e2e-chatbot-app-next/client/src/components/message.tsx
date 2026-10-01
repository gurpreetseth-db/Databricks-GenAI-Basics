import { motion } from 'framer-motion';
import React, { memo, useState } from 'react';
import { AnimatedAssistantIcon } from './animation-assistant-icon';
import { Response } from './elements/response';
import { MessageContent } from './elements/message';
import {
  Tool,
  ToolHeader,
  ToolContent,
  ToolInput,
  ToolOutput,
  type ToolState,
} from './elements/tool';
import {
  McpTool,
  McpToolHeader,
  McpToolContent,
  McpToolInput,
  McpApprovalActions,
} from './elements/mcp-tool';
import { MessageActions } from './message-actions';
import { PreviewAttachment } from './preview-attachment';
import equal from 'fast-deep-equal';
import { cn, sanitizeText } from '@/lib/utils';
import { MessageEditor } from './message-editor';
import { MessageReasoning } from './message-reasoning';
import type { UseChatHelpers } from '@ai-sdk/react';
import type { ChatMessage, Feedback } from '@chat-template/core';
import { useDataStream } from './data-stream-provider';
import {
  createMessagePartSegments,
  formatNamePart,
  isNamePart,
  joinMessagePartSegments,
} from './databricks-message-part-transformers';
import { MessageError } from './message-error';
import { MessageOAuthError } from './message-oauth-error';
import { isCredentialErrorMessage } from '@/lib/oauth-error-utils';
import { Streamdown } from 'streamdown';
import { useApproval } from '@/hooks/use-approval';
import { ChevronRight, ScrollText } from 'lucide-react';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from './ui/collapsible';

const PurePreviewMessage = ({
  message,
  allMessages,
  isLoading,
  setMessages,
  addToolApprovalResponse,
  sendMessage,
  regenerate,
  isReadonly,
  requiresScrollPadding,
  initialFeedback,
}: {
  message: ChatMessage;
  allMessages: ChatMessage[];
  isLoading: boolean;
  setMessages: UseChatHelpers<ChatMessage>['setMessages'];
  addToolApprovalResponse: UseChatHelpers<ChatMessage>['addToolApprovalResponse'];
  sendMessage: UseChatHelpers<ChatMessage>['sendMessage'];
  regenerate: UseChatHelpers<ChatMessage>['regenerate'];
  isReadonly: boolean;
  requiresScrollPadding: boolean;
  initialFeedback?: Feedback;
}) => {
  const [mode, setMode] = useState<'view' | 'edit'>('view');
  const [showErrors, setShowErrors] = useState(false);
  // Logs panel open state: null = follow isLoading (auto-open while the agent
  // works, auto-collapse once the final answer lands); true/false = user choice.
  const [logsOpen, setLogsOpen] = useState<boolean | null>(null);

  // Hook for handling MCP approval requests
  const { submitApproval, isSubmitting, pendingApprovalId } = useApproval({
    addToolApprovalResponse,
    sendMessage,
  });

  const attachmentsFromMessage = message.parts.filter(
    (part) => part.type === 'file',
  );

  // Extract non-OAuth error parts separately (OAuth errors are rendered inline)
  const errorParts = React.useMemo(
    () =>
      message.parts
        .filter((part) => part.type === 'data-error')
        .filter((part) => {
          // OAuth errors are rendered inline, not in the error section
          return !isCredentialErrorMessage(part.data);
        }),
    [message.parts],
  );

  useDataStream();

  const partSegments = React.useMemo(
    /**
     * We segment message parts into segments that can be rendered as a single component.
     * Used to render citations as part of the associated text.
     * Note: OAuth errors are included here for inline rendering, non-OAuth errors are filtered out.
     */
    () =>
      createMessagePartSegments(
        message.parts.filter(
          (part) =>
            part.type !== 'data-error' || isCredentialErrorMessage(part.data),
        ),
      ),
    [message.parts],
  );

  // Check if message only contains non-OAuth errors (no other content)
  const hasOnlyErrors = React.useMemo(() => {
    const nonErrorParts = message.parts.filter(
      (part) => part.type !== 'data-error',
    );
    // Only consider non-OAuth errors for this check
    return errorParts.length > 0 && nonErrorParts.length === 0;
  }, [message.parts, errorParts.length]);

  // Split rendered segments into two buckets:
  //  - mainEls: the final answer, citations, interactive MCP approval prompts and
  //    OAuth re-auth — everything the user should see directly in the chat.
  //  - logEls: reasoning + completed tool calls (parameters/results) — the
  //    "work", tucked into a collapsible Logs panel so the main window stays clean.
  const mainEls: React.ReactNode[] = [];
  const logEls: React.ReactNode[] = [];

  partSegments?.forEach((parts, index) => {
    const [part] = parts;
    const { type } = part;
    const key = `message-${message.id}-part-${index}`;

    if (type === 'reasoning' && part.text?.trim().length > 0) {
      logEls.push(
        <MessageReasoning key={key} isLoading={isLoading} reasoning={part.text} />,
      );
      return;
    }

    if (type === 'text') {
      if (isNamePart(part)) {
        mainEls.push(
          <Streamdown
            key={key}
            className="-mb-2 mt-0 border-l-4 pl-2 text-muted-foreground"
          >{`# ${formatNamePart(part)}`}</Streamdown>,
        );
        return;
      }
      if (mode === 'view') {
        mainEls.push(
          <div key={key}>
            <MessageContent
              data-testid="message-content"
              className={cn({
                'w-fit break-words rounded-2xl px-3 py-2 text-right text-white':
                  message.role === 'user',
                'bg-transparent px-0 py-0 text-left':
                  message.role === 'assistant',
              })}
              style={
                message.role === 'user'
                  ? { backgroundColor: '#006cff' }
                  : undefined
              }
            >
              <Response>
                {sanitizeText(joinMessagePartSegments(parts))}
              </Response>
            </MessageContent>
          </div>,
        );
        return;
      }
      if (mode === 'edit') {
        mainEls.push(
          <div key={key} className="flex w-full flex-row items-start gap-3">
            <div className="size-8" />
            <div className="min-w-0 flex-1">
              <MessageEditor
                key={message.id}
                message={message}
                setMode={setMode}
                setMessages={setMessages}
                regenerate={regenerate}
              />
            </div>
          </div>,
        );
        return;
      }
      return;
    }

    // Databricks tool calls and results
    if (part.type === `dynamic-tool`) {
      const { toolCallId, input, state, errorText, output, toolName } = part;

      // MCP tool call? (has approvalRequestId in metadata across all states)
      const isMcpApproval =
        part.callProviderMetadata?.databricks?.approvalRequestId != null;
      const mcpServerName =
        part.callProviderMetadata?.databricks?.mcpServerName?.toString();

      const approved: boolean | undefined =
        'approval' in part ? part.approval?.approved : undefined;

      const effectiveState: ToolState = (() => {
        if (
          part.providerExecuted &&
          !isLoading &&
          state === 'input-available'
        ) {
          return 'output-available';
        }
        return state;
      })();

      let el: React.ReactNode;
      if (isMcpApproval) {
        el = (
          <McpTool key={toolCallId} defaultOpen={true}>
            <McpToolHeader
              serverName={mcpServerName}
              toolName={toolName}
              state={effectiveState}
              approved={approved}
            />
            <McpToolContent>
              <McpToolInput input={input} />
              {state === 'approval-requested' && (
                <McpApprovalActions
                  onApprove={() =>
                    submitApproval({
                      approvalRequestId: toolCallId,
                      approve: true,
                    })
                  }
                  onDeny={() =>
                    submitApproval({
                      approvalRequestId: toolCallId,
                      approve: false,
                    })
                  }
                  isSubmitting={
                    isSubmitting && pendingApprovalId === toolCallId
                  }
                />
              )}
              {state === 'output-available' && output != null && (
                <ToolOutput
                  output={
                    errorText ? (
                      <div className="rounded border p-2 text-red-500">
                        Error: {errorText}
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap font-mono text-sm">
                        {typeof output === 'string'
                          ? output
                          : JSON.stringify(output, null, 2)}
                      </div>
                    )
                  }
                  errorText={undefined}
                />
              )}
            </McpToolContent>
          </McpTool>
        );
      } else {
        el = (
          <Tool key={toolCallId} defaultOpen={true}>
            <ToolHeader type={toolName} state={effectiveState} />
            <ToolContent>
              <ToolInput input={input} />
              {state === 'output-available' && (
                <ToolOutput
                  output={
                    errorText ? (
                      <div className="rounded border p-2 text-red-500">
                        Error: {errorText}
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap font-mono text-sm">
                        {typeof output === 'string'
                          ? output
                          : JSON.stringify(output, null, 2)}
                      </div>
                    )
                  }
                  errorText={undefined}
                />
              )}
            </ToolContent>
          </Tool>
        );
      }

      // Interactive MCP approval prompts must stay in the main flow so the user
      // can act on them; everything else (completed tool calls) goes to Logs.
      const isInteractive = isMcpApproval && state === 'approval-requested';
      (isInteractive ? mainEls : logEls).push(el);
      return;
    }

    // Citations / annotations
    if (type === 'source-url') {
      mainEls.push(
        <a
          key={key}
          href={part.url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-baseline text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
        >
          <sup className="text-xs">[{part.title || part.url}]</sup>
        </a>,
      );
      return;
    }

    // OAuth errors inline
    if (type === 'data-error' && isCredentialErrorMessage(part.data)) {
      mainEls.push(
        <MessageOAuthError
          key={key}
          error={part.data}
          allMessages={allMessages}
          setMessages={setMessages}
          sendMessage={sendMessage}
        />,
      );
      return;
    }
  });

  // Follow isLoading until the user explicitly toggles the Logs panel.
  const isLogsOpen = logsOpen ?? isLoading;

  return (
    <div
      data-testid={`message-${message.role}`}
      className="group/message w-full"
      data-role={message.role}
    >
      <div
        className={cn('flex w-full items-start gap-2 md:gap-3', {
          'justify-end': message.role === 'user',
          'justify-start': message.role === 'assistant',
        })}
      >
        {message.role === 'assistant' && (
          <AnimatedAssistantIcon size={14} isLoading={isLoading} />
        )}

        <div
          className={cn('flex min-w-0 flex-col gap-3', {
            'w-full': message.role === 'assistant' || mode === 'edit',
            'min-h-96': message.role === 'assistant' && requiresScrollPadding,
            'max-w-[70%] sm:max-w-[min(fit-content,80%)]':
              message.role === 'user' && mode !== 'edit',
          })}
        >
          {attachmentsFromMessage.length > 0 && (
            <div
              data-testid={`message-attachments`}
              className="flex flex-row justify-end gap-2"
            >
              {attachmentsFromMessage.map((attachment) => (
                <PreviewAttachment
                  key={attachment.url}
                  attachment={{
                    name: attachment.filename ?? 'file',
                    contentType: attachment.mediaType,
                    url: attachment.url,
                  }}
                />
              ))}
            </div>
          )}

          {/* Final answer, citations, interactive prompts, OAuth re-auth */}
          {mainEls}

          {/* Logs: reasoning + tool calls (parameters/results) */}
          {logEls.length > 0 && (
            <Collapsible
              open={isLogsOpen}
              onOpenChange={setLogsOpen}
              className="w-full"
            >
              <CollapsibleTrigger className="flex items-center gap-1.5 rounded-md px-2 py-1 text-muted-foreground text-xs hover:bg-muted hover:text-foreground">
                <ChevronRight
                  className={cn('h-3.5 w-3.5 transition-transform', {
                    'rotate-90': isLogsOpen,
                  })}
                />
                <ScrollText className="h-3.5 w-3.5" />
                <span className="font-medium">Logs</span>
                <span className="rounded-full bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                  {logEls.length} step{logEls.length === 1 ? '' : 's'}
                </span>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2 flex flex-col gap-2 border-muted border-l-2 pl-3">
                {logEls}
              </CollapsibleContent>
            </Collapsible>
          )}

          {!isReadonly && !hasOnlyErrors && (
            <MessageActions
              key={`action-${message.id}`}
              message={message}
              isLoading={isLoading}
              setMode={setMode}
              errorCount={errorParts.length}
              showErrors={showErrors}
              onToggleErrors={() => setShowErrors(!showErrors)}
              initialFeedback={initialFeedback}
            />
          )}

          {errorParts.length > 0 && (hasOnlyErrors || showErrors) && (
            <div className="flex flex-col gap-2">
              {errorParts.map((part, index) => (
                <MessageError
                  key={`error-${message.id}-${index}`}
                  error={part.data}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export const PreviewMessage = memo(
  PurePreviewMessage,
  (prevProps, nextProps) => {
    if (prevProps.isLoading !== nextProps.isLoading) return false;
    // While streaming, re-render whenever the AI SDK produces a new message
    // object (each throttled update). We use reference equality rather than
    // deep-equal on parts because fast-deep-equal short-circuits on identical
    // references — and the SDK may mutate parts in place during streaming.
    if (nextProps.isLoading && prevProps.message !== nextProps.message)
      return false;

    if (prevProps.message.id !== nextProps.message.id) return false;
    if (prevProps.requiresScrollPadding !== nextProps.requiresScrollPadding)
      return false;
    if (!equal(prevProps.message.parts, nextProps.message.parts)) return false;
    if (prevProps.initialFeedback?.feedbackType !== nextProps.initialFeedback?.feedbackType)
      return false;

    return true; // Props are equal, skip re-render
  },
);

export const AwaitingResponseMessage = () => {
  const role = 'assistant';

  return (
    <div
      data-testid="message-assistant-loading"
      className="group/message w-full"
      data-role={role}
    >
      <div className="flex items-start justify-start gap-3">
        <AnimatedAssistantIcon size={14} isLoading={false} muted={true} />

        <div className="flex w-full flex-col gap-2 md:gap-4">
          <div className="p-0 text-muted-foreground text-sm">
            <LoadingText>Thinking...</LoadingText>
          </div>
        </div>
      </div>
    </div>
  );
};

const LoadingText = ({ children }: { children: React.ReactNode }) => {
  return (
    <motion.div
      animate={{ backgroundPosition: ['100% 50%', '-100% 50%'] }}
      transition={{
        duration: 1.5,
        repeat: Number.POSITIVE_INFINITY,
        ease: 'linear',
      }}
      style={{
        background:
          'linear-gradient(90deg, hsl(var(--muted-foreground)) 0%, hsl(var(--muted-foreground)) 35%, hsl(var(--foreground)) 50%, hsl(var(--muted-foreground)) 65%, hsl(var(--muted-foreground)) 100%)',
        backgroundSize: '200% 100%',
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
      }}
      className="flex items-center text-transparent"
    >
      {children}
    </motion.div>
  );
};
