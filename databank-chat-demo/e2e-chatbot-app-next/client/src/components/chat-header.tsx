import { useNavigate } from 'react-router-dom';
import { useWindowSize } from 'usehooks-ts';

import { SidebarToggle } from '@/components/sidebar-toggle';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useSidebar } from './ui/sidebar';
import {
  PlusIcon,
  CloudOffIcon,
  MessageSquareOff,
  Landmark,
  Zap,
  Clock,
  BrainCircuit,
  type LucideIcon,
} from 'lucide-react';
import { useConfig } from '@/hooks/use-config';
import type { MemoryMode } from '@/contexts/AppConfigContext';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

const DOCS_URL =
  'https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app';

// Deployment-mode metadata: label + icon + badge styling + tooltip.
const MODE_META: Record<
  MemoryMode,
  { label: string; Icon: LucideIcon; variant: 'secondary' | 'default'; hint: string }
> = {
  simple: {
    label: 'Simple',
    Icon: Zap,
    variant: 'secondary',
    hint: 'Simple mode — conversations are ephemeral (no memory).',
  },
  shortterm: {
    label: 'Short-Term Memory',
    Icon: Clock,
    variant: 'secondary',
    hint: 'Short-term memory — history is kept for this browser session only.',
  },
  longterm: {
    label: 'Long-Term Memory',
    Icon: BrainCircuit,
    variant: 'default',
    hint: 'Long-term memory — history and learned facts persist across sessions.',
  },
};

export function ChatHeader() {
  const navigate = useNavigate();
  const { open } = useSidebar();
  const { chatHistoryEnabled, feedbackEnabled, memoryMode } = useConfig();

  const { width: windowWidth } = useWindowSize();
  const mode = MODE_META[memoryMode] ?? MODE_META.simple;
  const ModeIcon = mode.Icon;

  return (
    <header className="sticky top-0 z-10 flex items-center gap-2 border-b bg-background px-2 py-2 md:px-3">
      <SidebarToggle />

      {(!open || windowWidth < 768) && (
        <Button
          variant="outline"
          className="order-2 ml-auto h-8 px-2 md:order-1 md:ml-0 md:h-fit md:px-2"
          onClick={() => {
            navigate('/');
          }}
        >
          <PlusIcon />
          <span className="md:sr-only">New Chat</span>
        </Button>
      )}

      {/* DataBank brand / logo */}
      <button
        type="button"
        onClick={() => navigate('/')}
        className="order-1 flex items-center gap-2.5 md:order-2"
        aria-label="DataBank home"
      >
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-indigo-700 text-white shadow-sm">
          <Landmark className="h-5 w-5" />
        </span>
        <span className="flex flex-col leading-tight">
          <span className="font-semibold text-sm tracking-tight">
            DataBank
          </span>
          <span className="hidden text-[10px] text-muted-foreground sm:inline">
            AI Financial Assistant
          </span>
        </span>
      </button>

      <div className="order-3 ml-auto flex items-center gap-2">
        {/* Deployment-mode indicator (top-right) */}
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-1.5">
                <span className="hidden text-muted-foreground text-xs lg:inline">
                  Deployment Mode:
                </span>
                <Badge
                  variant={mode.variant}
                  className="flex items-center gap-1.5"
                >
                  <ModeIcon className="h-3 w-3" />
                  <span>{mode.label}</span>
                </Badge>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>{mode.hint}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {!chatHistoryEnabled && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <a
                  href={DOCS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 rounded-full bg-muted px-2 py-1 text-muted-foreground text-xs hover:text-foreground"
                >
                  <CloudOffIcon className="h-3 w-3" />
                  <span className="hidden sm:inline">Ephemeral</span>
                </a>
              </TooltipTrigger>
              <TooltipContent>
                <p>Chat history disabled — conversations are not saved. Click to learn more.</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
        {!feedbackEnabled && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <a
                  href={DOCS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 rounded-full bg-muted px-2 py-1 text-muted-foreground text-xs hover:text-foreground"
                >
                  <MessageSquareOff className="h-3 w-3" />
                  <span className="hidden sm:inline">Feedback disabled</span>
                </a>
              </TooltipTrigger>
              <TooltipContent>
                <p>Feedback submission disabled. Click to learn more.</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
    </header>
  );
}
