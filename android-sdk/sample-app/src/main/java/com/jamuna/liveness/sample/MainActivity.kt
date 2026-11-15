package com.jamuna.liveness.sample

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.jamuna.liveness.LivenessSdk
import com.jamuna.liveness.model.LivenessRequest
import com.jamuna.liveness.model.LivenessResult
import com.jamuna.liveness.sample.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private val livenessLauncher = registerForActivityResult(LivenessSdk.Contract()) { result ->
        val text = when (result) {
            is LivenessResult.Approved -> "APPROVED\n${result.reasonCodes}"
            is LivenessResult.Rejected -> "REJECTED\n${result.message}\n${result.reasonCodes}"
            is LivenessResult.Error -> "ERROR\n${result.message}"
            is LivenessResult.Cancelled -> "Cancelled"
        }
        binding.txtResult.text = text
        Toast.makeText(this, text.take(80), Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Defaults for pilot — prefer HTTPS domain when nginx proxies to :8001
        binding.inputBaseUrl.setText("https://your-liveness-api.example.com")
        binding.inputApiKey.setText("liveness-pilot-key")
        binding.inputAccount.setText("1234567890")

        binding.btnStart.setOnClickListener {
            val baseUrl = binding.inputBaseUrl.text?.toString()?.trim().orEmpty()
            val apiKey = binding.inputApiKey.text?.toString()?.trim().orEmpty()
            val account = binding.inputAccount.text?.toString()?.trim().orEmpty()
            if (baseUrl.isEmpty() || account.isEmpty()) {
                Toast.makeText(this, "Base URL and account number required", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            livenessLauncher.launch(
                LivenessRequest(
                    baseUrl = baseUrl,
                    apiKey = apiKey,
                    accountNumber = account,
                )
            )
        }
    }
}
