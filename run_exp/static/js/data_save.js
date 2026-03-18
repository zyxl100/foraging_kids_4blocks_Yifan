let save_final_deter
// var prompt_resubmit = function() {
// 		document.body.innerHTML = error_message;
// 		$("#resubmit").click(resubmit);
// 	};

async function markVersion1AsFinished() {
  const fetchUrl = "https://gwdeterversion.vercel.app/api/fetch-runsheet";
  const uploadUrl = "https://gwdeterversion.vercel.app/api/upload-runsheet";

  try {
    console.log("📥 Fetching runsheet...");
    const res = await fetch(fetchUrl);
    const rows = await res.json();
    console.log("📦 Current rows:", rows);

    const headers = Object.keys(rows[0]);

    // Find and mark first unfinished version 1 row
    const targetRow = rows.find(row => row.version === "1" && !row.finished);
    if (!targetRow) {
      console.warn("❌ No available version 1 row to mark as finished.");
      return;
    }

    targetRow.finished = "1";
    console.log("✅ Marked row as finished:", targetRow);

    // Convert back to CSV
    const csv = [headers.join(",")].concat(
      rows.map(row =>
        headers.map(h => `"${(row[h] || "").replace(/"/g, '""')}"`).join(",")
      )
    ).join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const formData = new FormData();
    formData.append("file", blob, "runsheet.csv");

    const uploadRes = await fetch(uploadUrl, {
      method: "POST",
      body: formData
    });

    if (!uploadRes.ok) {
      const errText = await uploadRes.text();
      console.error("❌ Upload failed:", errText);
    } else {
      console.log("✅ Runsheet updated and uploaded.");
    }

  } catch (err) {
    console.error("💥 Unexpected error:", err);
  }
}


function generateRandomIdentifier() {
    const now = new Date();
    const hour = now.getHours().toString().padStart(2, '0');
    const minute = now.getMinutes().toString().padStart(2, '0');
    const rand = Math.floor(Math.random() * 1000).toString().padStart(3, '0');
    
    const randomIdentifier = `${hour}${minute}_${rand}`;
    return randomIdentifier;
}

let randomidentifier;

// var resubmit = function() {
// 		document.body.innerHTML = "<h1>Trying to resubmit...</h1>";
// 		reprompt = setTimeout(prompt_resubmit, 10000);

// 		psiTurk.saveData({
// 			success: function() {
// 			    clearInterval(reprompt);
//           psiturk.completeHIT(); // when finished saving compute bonus, the quit
// 			},
// 			error: prompt_resubmit
// 		});
// 	};

  var save_data = function(final) {
  // exclude unwanted keys/columns
  // var exclude_keys = ['internal_node_id', 'trial_index','rt','stimulus','time_elapsed','responses'];
    var clean_data = jsPsych.data.get()
    // var callback = function() {
    //   // Only save data without completing the HIT during the task
    //   psiturk.saveData({
    //     success: function(){
    //       console.log("Data saved during the task.");
    //     },
    //     error: prompt_resubmit
    //   });
      
    //   // If final is true, we complete the HIT after task finishes
    //   if (final) {
    //     psiturk.completeHIT();  // Complete the HIT when task is done
    //   }
    // };

    // Call the callback to save the data during the task
    // callback();
  /* Save participant data file */

  // Set participant data file name
  if (debug_mode) {
    var data_file_name = "dev_test.csv";

  } else {
    if (save_final_deter=='final'){
      var data_file_name =  'final_S_' + useridtouse + '.csv';
    }else{
      randomidentifier = generateRandomIdentifier();
      var data_file_name =  'S_' + useridtouse +'_'+randomidentifier+ '.csv';
    }
  }

  // Save participant data file as a download in the web browser
  // Note that unlike saving server-side, this doesn't remove quotation marks from the CSV file
  if (data_save_method == 'csv_client') {
    clean_data.localSave('csv', data_file_name);
    // Save participant data file on a server side directory via PHP
    // (Broken: only works with Apache + PHP and no psiTurk)
  } else if(data_save_method == 'csv_server_php') {
    saveData(data_file_name, clean_data.csv())
    // Save participant data file on a server side directory via Python (only works with psiTurk)
  }  else if(data_save_method == 'csv_server_py') {
    var all_trials = jsPsych.data.get().values();

    // Filter out trials where ignore is true
    // var filtered_trials = all_trials
    //
    // Convert manually to CSV
    // function arrayToCSV(data) {
    //   if (data.length === 0) return '';

    //   // Get all unique keys across all rows
    //   const keys = [...new Set(data.flatMap(row => Object.keys(row)))];

    //   const csvRows = [
    //     keys.join(','), // header row
    //     ...data.map(row =>
    //       keys.map(k => JSON.stringify(row[k] ?? '')).join(',')
    //     )
    //   ];

    //   return csvRows.join('\n');
    // }
    // var clean_csv = arrayToCSV(filtered_trials);

    clean_csv = jsPsych.data.get().csv()
    // $.ajax({
    //   type: 'POST',
    //   url: "../save_data_file",
    //   dataType: 'json',
    //   success: callback,
    //   error: callback,
    //   data: {
    //     file_name: data_file_name,
    //     file_data: clean_csv,
    //   },
    // });
  }
}

