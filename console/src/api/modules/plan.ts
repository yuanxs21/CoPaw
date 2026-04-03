import { request } from "../request";
import { getApiToken, getApiUrl } from "../config";
import { buildAuthHeaders } from "../authHeaders";
import type {
  Plan,
  PlanConfig,
  RevisePlanRequest,
  FinishPlanRequest,
} from "../types";

export const planApi = {
  getCurrentPlan: () => request<Plan | null>("/plan/current"),

  revisePlan: (body: RevisePlanRequest) =>
    request<Plan>("/plan/revise", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  finishPlan: (body: FinishPlanRequest) =>
    request<{ success: boolean }>("/plan/finish", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getPlanConfig: () => request<PlanConfig>("/plan/config"),

  updatePlanConfig: (config: PlanConfig) =>
    request<PlanConfig>("/plan/config", {
      method: "PUT",
      body: JSON.stringify(config),
    }),

  confirmPlan: () =>
    request<{ confirmed: boolean; started_subtask_idx: number | null }>(
      "/plan/confirm",
      { method: "POST" },
    ),
};

/**
 * Subscribe to real-time plan updates via SSE.
 * When an API token is present, obtains a short-lived single-use ticket
 * via POST (Bearer header) so the long-lived token is not put in the URL.
 * Returns a Promise of an unsubscribe function that closes the EventSource.
 */
export async function subscribePlanUpdates(
  onUpdate: (plan: Plan | null) => void,
): Promise<() => void> {
  const url = getApiUrl("/plan/stream");
  const headers = buildAuthHeaders();
  const agentId = headers["X-Agent-Id"];

  const params = new URLSearchParams();
  if (getApiToken()) {
    const { ticket } = await request<{ ticket: string }>(
      "/plan/stream/ticket",
      { method: "POST" },
    );
    params.set("ticket", ticket);
  }
  if (agentId) params.set("agent_id", agentId);
  const sep = url.includes("?") ? "&" : "?";
  const fullUrl = params.toString() ? `${url}${sep}${params.toString()}` : url;

  const es = new EventSource(fullUrl);

  es.addEventListener("plan_update", (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      onUpdate(data);
    } catch {
      // ignore malformed events
    }
  });

  es.onerror = () => {
    // EventSource auto-reconnects; we just log
    console.warn("Plan SSE connection error — will auto-reconnect");
  };

  return () => es.close();
}
