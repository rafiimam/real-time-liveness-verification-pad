package com.jamuna.liveness.net

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Client for Decision API `WS /ws/live` protocol.
 */
internal class LiveWsClient(
    private val baseUrl: String,
    private val apiKey: String,
    private val listener: Listener,
) {
    interface Listener {
        fun onReady(sessionId: String, nonce: String, earlySeconds: Double, maxSeconds: Double)
        fun onProgress(frames: Int, canFinish: Boolean, elapsedMs: Long)
        fun onStatus(message: String)
        fun onResult(json: JSONObject)
        fun onError(message: String)
    }

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()

    private var socket: WebSocket? = null
    private val closed = AtomicBoolean(false)
    private var nonce: String? = null

    fun connectAndStart(
        accountNumber: String,
        photoJpegBase64: String,
        transactionAmount: Double? = null,
        transactionId: String? = null,
    ) {
        val wsUrl = toWsUrl(baseUrl) + "/ws/live?access_token=" + apiKey
        val request = Request.Builder()
            .url(wsUrl)
            .header("Authorization", "Bearer $apiKey")
            .build()

        socket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                val start = JSONObject()
                    .put("type", "start")
                    .put("account_number", accountNumber)
                    // Backward-compatible field for older API builds
                    .put("account_id", accountNumber)
                    .put("photo_base64", photoJpegBase64)
                    .put("access_token", apiKey)
                    .put("sdk_version", com.jamuna.liveness.BuildConfig.SDK_VERSION)
                    .put("platform", "android")
                if (transactionAmount != null) start.put("transaction_amount", transactionAmount)
                if (!transactionId.isNullOrBlank()) start.put("transaction_id", transactionId)
                webSocket.send(start.toString())
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                if (closed.get()) return
                val msg = try {
                    JSONObject(text)
                } catch (_: Exception) {
                    return
                }
                when (msg.optString("type")) {
                    "ready" -> {
                        nonce = msg.optString("nonce")
                        listener.onReady(
                            sessionId = msg.optString("session_id"),
                            nonce = nonce.orEmpty(),
                            earlySeconds = msg.optDouble("early_seconds", 4.0),
                            maxSeconds = msg.optDouble("max_seconds", 10.0),
                        )
                    }
                    "progress" -> listener.onProgress(
                        frames = msg.optInt("frames"),
                        canFinish = msg.optBoolean("can_finish"),
                        elapsedMs = msg.optLong("elapsed_ms"),
                    )
                    "status" -> listener.onStatus(msg.optString("message", "Checking…"))
                    "result" -> {
                        closed.set(true)
                        listener.onResult(msg)
                        closeQuietly()
                    }
                    "error" -> {
                        closed.set(true)
                        listener.onError(msg.optString("message", "Unknown error"))
                        closeQuietly()
                    }
                    "pong" -> Unit
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                if (closed.getAndSet(true)) return
                listener.onError(t.message ?: "WebSocket failure")
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                if (closed.getAndSet(true)) return
                listener.onError(reason.ifBlank { "WebSocket closed ($code)" })
            }
        })
    }

    fun sendFrame(jpegBase64DataUrlOrRaw: String, ts: Long = System.currentTimeMillis()) {
        val n = nonce ?: return
        val payload = JSONObject()
            .put("type", "frame")
            .put("image_base64", jpegBase64DataUrlOrRaw)
            .put("ts", ts)
            .put("nonce", n)
        socket?.send(payload.toString())
    }

    fun sendEnd() {
        val n = nonce ?: return
        val payload = JSONObject()
            .put("type", "end")
            .put("nonce", n)
        socket?.send(payload.toString())
    }

    fun closeQuietly() {
        closed.set(true)
        try {
            socket?.close(1000, "done")
        } catch (_: Exception) {
        }
        socket = null
    }

    companion object {
        fun toWsUrl(httpBase: String): String {
            val trimmed = httpBase.trim().trimEnd('/')
            return when {
                trimmed.startsWith("https://") -> "wss://" + trimmed.removePrefix("https://")
                trimmed.startsWith("http://") -> "ws://" + trimmed.removePrefix("http://")
                else -> trimmed
            }
        }
    }
}
