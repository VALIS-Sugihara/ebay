var results;

//CSVファイルを読み込む関数getCSV()の定義
function getCSV(){
    var now = new Date();
    var yyyymmdd = now.getFullYear()+
      ( "0"+( now.getMonth()+1 ) ).slice(-2)+
      ( "0"+now.getDate() ).slice(-2);
    var req = new XMLHttpRequest(); // HTTPでファイルを読み込むためのXMLHttpRrequestオブジェクトを生成
    req.open("get", "http://ebay-frontend.s3-website-ap-northeast-1.amazonaws.com/data/ebay_yahoo_detail_20191001.csv", true); // アクセスするファイルを指定
    // req.open("get", "http://ebay-frontend.s3-website-ap-northeast-1.amazonaws.com/data/ebay_yahoo_detail_"+yyyymmdd+".csv", true); // アクセスするファイルを指定
    req.send(null); // HTTPリクエストの発行

    // レスポンスが返ってきたらconvertCSVtoArray()を呼ぶ
    req.onload = function(){
    	results = convertCSVtoArray(req.responseText); // 渡されるのは読み込んだCSVデータ
    	console.log(results[0])
    	columns = results[0]
    	// app.data["results"] = results
      var app = new Vue({
          el: '#app',
          data: {
              message: "Hello Vue js.",
              ebay: "ebay",
              yahoo: "yahoo",
              results: results
          }
      })
    }
}

// 読み込んだCSVデータを二次元配列に変換する関数convertCSVtoArray()の定義
function convertCSVtoArray(str){ // 読み込んだCSVデータが文字列として渡される
    var results = []; // 最終的な二次元配列を入れるための配列
    var tmp = str.split("\n"); // 改行を区切り文字として行を要素とした配列を生成

    // 各行ごとにカンマで区切った文字列を要素とした二次元配列を生成
    for(var i=0;i<tmp.length;++i){
        results[i] = tmp[i].split(',');
    }
    return results
}

getCSV(); //最初に実行される
