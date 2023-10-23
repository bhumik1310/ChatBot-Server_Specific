import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const sanitizeFilename = require('sanitize-filename');
const path = require('path');
const url = require('url');
import { JSDOM } from 'jsdom';
const request = require('request');
const tldextract = require('tld-extract') 



//Function that downloads PDF from a URL , and saves it to a specified Path.
function downloadPDF(pdfUrl, savePath) {
  const options = {
    uri: pdfUrl,
    headers: {
      'User-Agent': 'Mozilla/5.0', // Set a User-Agent to avoid potential issues with some websites
    },
  };

  request(options)
    .pipe(fs.createWriteStream(savePath))
    .on('close', () => {
      console.log('PDF file has been downloaded and saved to:', savePath);
    })
    .on('error', (err) => {
      console.error('Error downloading the PDF:', err);
    });
}

// Function that reads the directory for files and return an array with a list of files to iterate over
function readDir(directory){
try {
  arrayOfFiles = fs.readdirSync(directory)       //./storage/datasets/default <- Default for reading JSON files in the folder
  // console.log(arrayOfFiles)
  return arrayOfFiles
} catch(e) {
  console.log(e)
}
}


//Function that checks if a folder is present, if not , it creates the folder.

function is_directory_present(url,subdir){
  const parsedUrl = new URL(url);  // Make and URL object out of the link and check if the folder of link is already present.
  const domain = parsedUrl.hostname
  domain.replace(/.+\/\/|www.|\..+/g,'');
  const folderPath = domain
  //pass
  var totalPath = "./CrawledData/"+folderPath+subdir
  //Check if the url has been crawled already or not (Check domain name folder)
  if (!fs.existsSync(totalPath)) {
    // If it doesn't exist, create the folder
    fs.mkdirSync(totalPath);
    console.log(`Folder '${folderPath}' created.`);
    if_dir_made=1
  } else { 
    console.log(`Folder '${folderPath}' already exists.`);
  }

return totalPath

}




// Cheerio crawler instantiation 

import { Dataset,CheerioCrawler } from 'crawlee';
import { dir } from 'console';
const fs = require('fs');
const csv = require('csv-parser');

const crawler = new CheerioCrawler({
    async requestHandler({ $,request,body }) {
      // Check if the link is of a PDF or regular link
      if ((request.url).includes('.pdf')|| (request.url).includes('.PDF')){

        //Create folder with name of url
        is_directory_present(request.url,'')
        
        //Create a seperate folder for PDFs inside the url folder and extract the totalPath of it
        var totalPath = is_directory_present(request.url,'/PDF/')

        // Calling the function to download the PDF and save it to the Folder.
        var url = request.url;
        
        downloadPDF(url,totalPath+sanitizeFilename(url));


 



      }
       else{ const title = $('title').text();

        function cleanString(input) {

            return input.replace(/<header[^>]*>[\s\S]*?<\/header>/gi, '')
           .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, '')
           .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, '')
           .replace(/<header-class[^>]*>[\s\S]*?<\/header-class>/gi, '')
           .replace(/<footer-class[^>]*>[\s\S]*?<\/footer-class>/gi, '')
           .replace(/<main-menu[^>]*>[\s\S]*?<\/main-menu>/gi, '')
           .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
           .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
           .replace(/<[^>]*>/g, '')
           .replace(/\s+/g, ' ');
         
        }
        var dom = new JSDOM(body);
        var doc = dom.window.document;
        var bod = doc.querySelector('body');
        var data = bod.textContent;

        console.log(`The title: ${title}`);
        await Dataset.pushData({
            url:request.url,
            // page_content: cleanString(body)
            content:data
        })}
    },
    additionalMimeTypes:['application/pdf']
})



const csvFilePath = './storage/key_value_stores/my-data/OUTPUT.csv';

const columnValues = [];

fs.createReadStream(csvFilePath)
  .pipe(csv())
  .on('data', (row) => {
    const value = row[Object.keys(row)[0]];
    columnValues.push(value);
  })
  .on('end', () => {
    // console.log('Column values:', columnValues);
    // console.log('Type of Thing', typeof columnValues);
    // console.log('At index 0 (before): \n', columnValues[0])
    // // columnValues.splice(0,1);
    // console.log('At index 0 (after): \n', columnValues[0])

  })
  .on('error', (error) => {
    console.error('Error reading CSV:', error);
  });


// Start the crawler with the provided URLs
await crawler.run(columnValues);
// Crawler saves files temporarily as json in datasets



// Check if the folder exists


// Recieving an array of JSON files
var arrayOfFiles=readDir("./storage/datasets/default");
try {
  arrayOfFiles = fs.readdirSync("./storage/datasets/default")
  // console.log(arrayOfFiles)
} catch(e) {
  console.log(e)
}


  // Iterating over each file in a folder
  for (const element of arrayOfFiles){
    var filePath="./storage/datasets/default/"+element;
    // Read the JSON file
    fs.readFile(filePath, 'utf8', (err, data) => {
            if (err) {
        console.error('Error reading the JSON file:', err);
        return;
    }

    try {
        // Parse the JSON data
        const jsonData = JSON.parse(data);

        // Create a string with the data you want to append
        let appendData = '\n';
        for (const key in jsonData) {
            if (jsonData.hasOwnProperty(key)) {
                const value = jsonData[key];
                appendData += `${key}: ${value}\n`;
            }
        }
        
        // console.log(typeof appendData);
        const txt_filename = jsonData['url'] + '.txt';
        // console.log(txt_filename)
        const totalPath = is_directory_present(jsonData['url'],'')
        const filename = sanitizeFilename(txt_filename)
        const filePath = path.join(totalPath,filename);
        fs.writeFile(filePath, appendData, (err) => {
          if (err) {
            console.error('Error writing the file:', err);
          } else {
            console.log(`File '${filename}' has been written to '${totalPath}}'.`);
          }
        });
        // console.log(sanitizeFilename(txt_filename));
    
    } catch (parseError) {
        console.error('Error parsing the JSON data:', parseError);
    }
});
  }

  // Iterating for PDF Files

  // var arrayPDF=[]
  // try: {}

