package com.jamuna.liveness.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

/**
 * @param baseUrl Decision API origin, e.g. `https://host` or `http://192.168.x.x:8001` (no trailing slash).
 * @param apiKey Bearer token / API key configured on the Decision API.
 * @param accountNumber Customer account number — backend resolves to NID + customer photo automatically.
 */
@Parcelize
data class LivenessRequest(
    val baseUrl: String,
    val apiKey: String,
    val accountNumber: String,
    val transactionAmount: Double? = null,
    val transactionId: String? = null,
) : Parcelable

@Parcelize
sealed class LivenessResult : Parcelable {
    @Parcelize
    data class Approved(
        val sessionId: String?,
        val faceMatch: Boolean?,
        val liveness: Boolean?,
        val reasonCodes: List<String>,
        val captureTrust: String?,
        val rawJson: String?,
        val nidPhotoMatch: Boolean? = null,
        val accountPhotoMatch: Boolean? = null,
        val nidVerified: Boolean? = null,
        val customerName: String? = null,
        val faceMatchProvider: String? = null,
        val cbsFacesDetected: Int? = null,
    ) : LivenessResult()

    @Parcelize
    data class Rejected(
        val sessionId: String?,
        val faceMatch: Boolean?,
        val liveness: Boolean?,
        val reasonCodes: List<String>,
        val captureTrust: String?,
        val message: String?,
        val rawJson: String?,
        val nidPhotoMatch: Boolean? = null,
        val accountPhotoMatch: Boolean? = null,
        val nidVerified: Boolean? = null,
        val customerName: String? = null,
        val faceMatchProvider: String? = null,
        val cbsFacesDetected: Int? = null,
    ) : LivenessResult()

    @Parcelize
    data class Error(val message: String) : LivenessResult()

    @Parcelize
    data object Cancelled : LivenessResult()
}
