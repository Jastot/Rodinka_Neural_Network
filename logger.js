function log(status, message){
  let dateObj = new Date();
  let date = `[${dateObj.getUTCHours()}:${dateObj.getUTCMinutes()}:${dateObj.getUTCSeconds()}]`;
  if (status == 0){
    console.log("\x1b[1m\x1b[32m%s\x1b[0m", `${date}[INF] ${message}`);
  } else if (status == 1){
    console.log("\x1b[1m\x1b[33m%s\x1b[0m", `${date}[WARN] ${message}`);
  } else if (status == 2){
    console.log("\x1b[4m\x1b[31m%s\x1b[0m", `${date}[ERR] ${message}`);
  } else {
    console.log("\x1b[0m%s\x1b[0m",`${date} ${message}`);
  }
}

//Exports
exports.log=log;