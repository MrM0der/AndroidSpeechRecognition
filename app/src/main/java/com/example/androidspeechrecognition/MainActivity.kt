package com.example.androidspeechrecognition
import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.ui.AppBarConfiguration
import android.view.Menu
import android.view.MenuItem
import androidx.core.app.ActivityCompat
import com.example.androidspeechrecognition.databinding.ActivityMainBinding
import java.io.File

private const val REQUEST_RECORD_AUDIO_PERMISSION = 200
private const val SAMPLE_RATE = 16_000
private const val MAX_SAMPLES = 3 * SAMPLE_RATE

class MainActivity : AppCompatActivity() {

    private var permissionToRecordAccepted = false
    private var permissions: Array<String> = arrayOf(Manifest.permission.RECORD_AUDIO)

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding

    private var record: AudioRecord? = null
    private var samples: FloatArray = FloatArray(MAX_SAMPLES)
    private var samplesRead: Int = 0

    private lateinit var speechRecognizer: SpeechRecognizer

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)

        speechRecognizer = SpeechRecognizer(File(filesDir, "wav2vec2-quant.basic.ort").path)
        binding.fab.setOnClickListener { view ->
//            Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
//                    .setAction("Action", null).show()
            if (record == null) {
                startRecording()
            } else {
                stopRecording()
                recognize()
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        permissionToRecordAccepted = if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        } else false
        if (!permissionToRecordAccepted) finish()
    }

    private fun startRecording() {
        record = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.MIC)
            .setAudioFormat(AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                            .setSampleRate(SAMPLE_RATE)
                            .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                            .build())
            .setBufferSizeInBytes(MAX_SAMPLES * 4)
            .build()

        record!!.startRecording()
        Log.i("ASR", "Recording started: " + record.toString());
        binding.contentMain.textViewLog.text = "Recording ..."
    }

    private fun stopRecording() {
        Log.i("ASR", "Try to stop " + record.toString())
        record!!.apply {
            stop()
            read(samples, 0, MAX_SAMPLES, AudioRecord.READ_BLOCKING)
                .also { samplesRead = it }
            Log.i("ASR", samplesRead.toString())
            Log.i("ASR", samples.minOrNull().toString() + ", " + samples.maxOrNull().toString())
            binding.contentMain.textViewLog.text = "Recording finished."
            release()
        }
        record = null
    }

    private fun recognize() {
        binding.contentMain.textViewLog.text = "Inferring ..."
        val sr = speechRecognizer.infer(samples.sliceArray(0 until samplesRead));
        binding.contentMain.textViewLog.text = "Result:"
        binding.contentMain.textViewRecognized.text = sr;
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        return when (item.itemId) {
            R.id.action_settings -> true
            else -> super.onOptionsItemSelected(item)
        }
    }
}
