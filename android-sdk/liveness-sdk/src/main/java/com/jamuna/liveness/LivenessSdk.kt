package com.jamuna.liveness

import android.app.Activity
import android.content.Context
import android.content.Intent
import androidx.activity.result.contract.ActivityResultContract
import com.jamuna.liveness.model.LivenessRequest
import com.jamuna.liveness.model.LivenessResult
import com.jamuna.liveness.ui.LivenessActivity

/**
 * Entry point for host Android apps.
 *
 * ```
 * val launcher = registerForActivityResult(LivenessSdk.Contract()) { result ->
 *   when (result) {
 *     is LivenessResult.Approved -> ...
 *     is LivenessResult.Rejected -> ...
 *     is LivenessResult.Error -> ...
 *     is LivenessResult.Cancelled -> ...
 *   }
 * }
 * launcher.launch(
 *   LivenessRequest(
 *     baseUrl = "https://liveness.example.com",
 *     apiKey = "liveness-pilot-key",
 *     accountNumber = "1234567890",
 *   )
 * )
 * ```
 */
object LivenessSdk {
    const val VERSION = BuildConfig.SDK_VERSION

    const val EXTRA_REQUEST = "com.jamuna.liveness.EXTRA_REQUEST"
    const val EXTRA_RESULT = "com.jamuna.liveness.EXTRA_RESULT"

    class Contract : ActivityResultContract<LivenessRequest, LivenessResult>() {
        override fun createIntent(context: Context, input: LivenessRequest): Intent {
            return Intent(context, LivenessActivity::class.java).putExtra(EXTRA_REQUEST, input)
        }

        override fun parseResult(resultCode: Int, intent: Intent?): LivenessResult {
            if (resultCode == Activity.RESULT_CANCELED) {
                return LivenessResult.Cancelled
            }
            @Suppress("DEPRECATION")
            val parcel = if (android.os.Build.VERSION.SDK_INT >= 33) {
                intent?.getParcelableExtra(EXTRA_RESULT, LivenessResult::class.java)
            } else {
                intent?.getParcelableExtra(EXTRA_RESULT)
            }
            return parcel ?: LivenessResult.Error("No result returned from LivenessActivity")
        }
    }
}
